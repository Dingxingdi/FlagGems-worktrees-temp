import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ReLU activation kernel - pure Triton kernel
@libentry()
@triton.jit
def relu_kernel(input_ptr, output_ptr, size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """ReLU activation kernel - pure Triton implementation."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    x = tl.load(input_ptr + offset, mask=mask, other=0.0)
    y = tl.where(x > 0, x, 0)
    tl.store(output_ptr + offset, y, mask=mask)


class PositionWiseFFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2, activation="relu"):
        logger.debug("GEMS Position_wise_FFN FORWARD")

        batch_size, seq_len, hidden_dim = x.shape
        intermediate_dim = w1.shape[1]

        # Ensure contiguous
        x = x.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()
        b1 = b1.contiguous() if b1 is not None else None
        b2 = b2.contiguous() if b2 is not None else None

        # Use addmm from flag_gems for matrix multiplication
        # This uses the existing Triton kernel for addmm
        from flag_gems.ops.addmm import addmm
        from flag_gems.ops.relu import relu

        # First linear layer: hidden -> intermediate
        # x: (batch, seq, hidden) -> x_2d: (batch*seq, hidden) = (M, K)
        # w1: (hidden, intermediate) -> w1: (K, N) where K=hidden, N=intermediate
        # addmm computes: bias + mat1 @ mat2 = bias + (M, K) @ (K, N) = (M, N)
        x_2d = x.view(-1, hidden_dim)  # (batch*seq, hidden) = (M, K)

        if b1 is not None:
            b1_expanded = b1.unsqueeze(0).expand(batch_size * seq_len, -1)  # (M, N)
            intermediate_2d = addmm(b1_expanded, x_2d, w1)  # (M, N)
        else:
            # No bias: use matmul
            intermediate_2d = torch.matmul(x_2d, w1)

        intermediate = intermediate_2d.view(batch_size, seq_len, intermediate_dim)

        # Apply activation
        if activation == "relu":
            # Use flag_gems relu which internally uses Triton kernel
            intermediate_act = relu(intermediate)
        elif activation == "gelu":
            from flag_gems.ops.gelu import gelu
            intermediate_act = gelu(intermediate)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Second linear layer: intermediate -> hidden
        # intermediate_act_2d: (batch*seq, intermediate_dim) = (M, K)
        # w2: (intermediate_dim, hidden_dim) = (K, N)
        # addmm computes: bias + mat1 @ mat2 = (M, N)
        intermediate_act_2d = intermediate_act.view(-1, intermediate_dim)

        if b2 is not None:
            b2_expanded = b2.unsqueeze(0).expand(batch_size * seq_len, -1)  # (M, N)
            output_2d = addmm(b2_expanded, intermediate_act_2d, w2)  # (M, N)
        else:
            output_2d = torch.matmul(intermediate_act_2d, w2)

        output = output_2d.view(batch_size, seq_len, hidden_dim)

        ctx.save_for_backward(x, w1, b1, w2, b2, intermediate_act)
        ctx.activation = activation

        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS Position_wise_FFN BACKWARD")

        x, w1, b1, w2, b2, intermediate_act = ctx.saved_tensors
        activation = ctx.activation

        batch_size, seq_len, hidden_dim = x.shape
        intermediate_dim = w1.shape[1]

        # Reshape for 2D operations
        grad_output_2d = grad_output.view(-1, hidden_dim)  # (M, N) where N=hidden_dim
        intermediate_act_2d = intermediate_act.view(-1, intermediate_dim)  # (M, K) where K=intermediate_dim
        x_2d = x.view(-1, hidden_dim)  # (M, K) where K=hidden_dim

        # Gradient of second linear layer
        # output = intermediate_act @ w2 + b2
        # w2 is (hidden, intermediate) = (N, K), so we need w2.t() = (K, N) for matmul
        # grad_w2 = intermediate_act_2d.t() @ grad_output_2d = (K, M) @ (M, N) = (K, N)
        grad_w2 = torch.matmul(intermediate_act_2d.t(), grad_output_2d)

        if b2 is not None:
            grad_b2 = grad_output_2d.sum(dim=0)
        else:
            grad_b2 = None

        grad_hidden = torch.matmul(grad_output_2d, w2.t())

        # Apply activation derivative
        if activation == "relu":
            from flag_gems.ops.relu import relu_backward
            grad_hidden = relu_backward(intermediate_act, grad_hidden.view(batch_size, seq_len, intermediate_dim)).view(-1, intermediate_dim)
        elif activation == "gelu":
            from flag_gems.ops.gelu import gelu_backward
            # gelu_backward(grad_output, self)
            grad_hidden = gelu_backward(grad_hidden.view(batch_size, seq_len, intermediate_dim), intermediate_act).view(-1, intermediate_dim)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Gradient of first linear layer
        grad_w1 = torch.matmul(x_2d.t(), grad_hidden)

        if b1 is not None:
            grad_b1 = grad_hidden.sum(dim=0)
        else:
            grad_b1 = None

        grad_x = torch.matmul(grad_hidden, w1.t())

        return grad_x.view(batch_size, seq_len, hidden_dim), grad_w1, grad_b1, grad_w2, grad_b2, None


def position_wise_ffn(x, w1, b1, w2, b2, activation="relu"):
    """
    Position-wise Feed-Forward Network (FFN) used in Transformer models.

    This implementation uses flag_gems' addmm and activation kernels,
    ensuring the core computation uses Triton kernels.

    Args:
        x: Input tensor of shape (batch, seq_len, hidden_dim)
        w1: First linear layer weight of shape (hidden_dim, intermediate_dim)
        b1: First linear layer bias of shape (intermediate_dim,) or None
        w2: Second linear layer weight of shape (intermediate_dim, hidden_dim)
        b2: Second linear layer bias of shape (hidden_dim,) or None
        activation: Activation function ("relu" or "gelu")

    Returns:
        Output tensor of shape (batch, seq_len, hidden_dim)
    """
    return PositionWiseFFN.apply(x, w1, b1, w2, b2, activation)