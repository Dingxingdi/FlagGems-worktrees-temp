import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=4, num_warps=4),
    ],
    key=["seq_len", "batch_size", "input_size"],
)
@triton.jit
def rnn_relu_forward_kernel(
    input_ptr,
    hx_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    output_ptr,
    hidden_output_ptr,
    seq_len,
    batch_size,
    input_size,
    hidden_size,
    num_layers,
    bidirectional,
    batch_first,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for rnn_relu forward pass.

    Supports:
    - Single layer (num_layers=1)
    - Unidirectional (bidirectional=False)
    - batch_first=False

    For more complex configurations, falls back to PyTorch implementation.
    """
    # Get program ID
    pid = tl.program_id(0)

    # Calculate output dimensions
    # output: (seq_len, batch, hidden_size) or (batch, seq_len, hidden_size)
    # hidden: (num_layers, batch, hidden_size)
    num_directions = 2 if bidirectional else 1
    total_seq = seq_len * batch_size

    # Each program processes one element
    if pid * BLOCK_SIZE >= total_seq:
        return

    # Calculate indices
    if batch_first:
        # input: (batch, seq_len, input_size)
        batch_idx = pid * BLOCK_SIZE // seq_len
        seq_idx = pid * BLOCK_SIZE % seq_len
        input_offset = batch_idx * seq_len * input_size + seq_idx * input_size
    else:
        # input: (seq_len, batch, input_size)
        seq_idx = pid * BLOCK_SIZE // batch_size
        batch_idx = pid * BLOCK_SIZE % batch_size
        input_offset = seq_idx * batch_size * input_size + batch_idx * input_size

    # Get output offset
    if batch_first:
        output_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size
    else:
        output_offset = seq_idx * batch_size * hidden_size + batch_idx * hidden_size

    # Load input for this time step
    offs_input = tl.arange(0, BLOCK_SIZE) * input_size
    mask_input = (tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE) < total_seq
    x = tl.load(input_ptr + input_offset + offs_input, mask=mask_input, other=0.0).to(tl.float32)

    # Initialize hidden state
    if hx_ptr is not None:
        hx_offset = batch_idx * hidden_size
        h = tl.load(hx_ptr + hx_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < hidden_size, other=0.0).to(tl.float32)
    else:
        h = tl.zeros(hidden_size, tl.float32)

    # Process each time step
    for t in range(seq_len):
        # Get input at time t
        if batch_first:
            t_input_offset = batch_idx * seq_len * input_size + t * input_size
        else:
            t_input_offset = t * batch_size * input_size + batch_idx * input_size

        x_t = tl.load(input_ptr + t_input_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < input_size, other=0.0).to(tl.float32)

        # Compute input-to-hidden: W_ih @ x_t + b_ih
        ih = tl.zeros(hidden_size, tl.float32)
        for i in range(0, input_size, 64):
            w_ih_chunk = tl.load(weight_ih_ptr + i * hidden_size + tl.arange(0, 64), mask=tl.arange(0, 64) < (input_size - i), other=0.0).to(tl.float32)
            x_chunk = tl.load(input_ptr + t_input_offset + i + tl.arange(0, 64), mask=tl.arange(0, 64) < (input_size - i), other=0.0).to(tl.float32)
            ih += tl.dot(x_chunk, w_ih_chunk)

        if bias_ih_ptr is not None:
            b_ih = tl.load(bias_ih_ptr + tl.arange(0, 64), mask=tl.arange(0, 64) < hidden_size, other=0.0).to(tl.float32)
            ih += b_ih

        # Compute hidden-to-hidden: W_hh @ h + b_hh
        hh = tl.zeros(hidden_size, tl.float32)
        for i in range(0, hidden_size, 64):
            w_hh_chunk = tl.load(weight_hh_ptr + i * hidden_size + tl.arange(0, 64), mask=tl.arange(0, 64) < (hidden_size - i), other=0.0).to(tl.float32)
            h_chunk = h
            hh += tl.dot(h_chunk, w_hh_chunk)

        if bias_hh_ptr is not None:
            b_hh = tl.load(bias_hh_ptr + tl.arange(0, 64), mask=tl.arange(0, 64) < hidden_size, other=0.0).to(tl.float32)
            hh += b_hh

        # Compute new hidden state: h = relu(ih + hh)
        h_new = ih + hh
        h = tl.where(h_new > 0, h_new, 0.0)

    # Store output
    if t == seq_len - 1:
        tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE), h, mask=tl.arange(0, BLOCK_SIZE) < hidden_size)

    # Store final hidden state
    if hidden_output_ptr is not None:
        hx_output_offset = batch_idx * hidden_size
        tl.store(hidden_output_ptr + hx_output_offset + tl.arange(0, BLOCK_SIZE), h, mask=tl.arange(0, BLOCK_SIZE) < hidden_size)


def rnn_relu_forward(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    """
    Forward pass for rnn_relu.

    Args:
        input: Tensor of shape (seq_len, batch, input_size) or (batch, seq_len, input_size)
        hx: Tensor of shape (num_layers * num_directions, batch, hidden_size)
        params: Tuple of weight tensors
        has_biases: Whether to use biases
        num_layers: Number of RNN layers
        dropout: Dropout probability
        train: Whether in training mode
        bidirectional: Whether to use bidirectional RNN
        batch_first: Whether batch is the first dimension

    Returns:
        output: Tensor of shape (seq_len, batch, hidden_size * num_directions) or (batch, seq_len, hidden_size * num_directions)
        hidden: Tensor of shape (num_layers * num_directions, batch, hidden_size)
    """
    logger.debug("GEMS RNN_RELU FORWARD")

    # Handle complex cases by falling back to PyTorch
    if num_layers > 1 or bidirectional:
        logger.debug("GEMS RNN_RELU: Falling back to PyTorch for multi-layer or bidirectional")
        return torch.rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)

    if dropout > 0 and train:
        logger.debug("GEMS RNN_RELU: Falling back to PyTorch for dropout in train mode")
        return torch.rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)

    # Get dimensions
    if batch_first:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input_size = input.shape[2]
    else:
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        input_size = input.shape[2]

    # Get hidden size from hx or params
    if hx is not None:
        hidden_size = hx.shape[2]
    else:
        # Try to get from params
        if len(params) > 0:
            hidden_size = params[0].shape[0]
        else:
            raise ValueError("Cannot determine hidden_size")

    # Determine output dimensions
    num_directions = 2 if bidirectional else 1
    if batch_first:
        output_shape = (batch_size, seq_len, hidden_size * num_directions)
    else:
        output_shape = (seq_len, batch_size, hidden_size * num_directions)

    # Create output tensors
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    hidden_shape = (num_layers * num_directions, batch_size, hidden_size)
    hidden = torch.empty(hidden_shape, dtype=input.dtype, device=input.device)

    # Handle hx being None
    if hx is None:
        hx = torch.zeros(hidden_shape, dtype=input.dtype, device=input.device)

    # Extract params
    if has_biases:
        weight_ih, bias_ih, weight_hh, bias_hh = params[:4]
    else:
        weight_ih = params[0]
        weight_hh = params[1]
        bias_ih = None
        bias_hh = None

    # Ensure contiguous
    input = input.contiguous()
    output = output.contiguous()
    hidden = hidden.contiguous()

    # Define grid
    total_elements = seq_len * batch_size

    # Calculate grid
    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    with torch.cuda.device(input.device):
        rnn_relu_forward_kernel[grid](
            input,
            hx,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            output,
            hidden,
            seq_len,
            batch_size,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            batch_first,
        )

    return output, hidden


class RnnReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first):
        logger.debug("GEMS RNN_RELU FORWARD")

        # Save for backward
        ctx.save_for_backward(input, hx)
        ctx.params = params
        ctx.has_biases = has_biases
        ctx.num_layers = num_layers
        ctx.dropout = dropout
        ctx.train = train
        ctx.bidirectional = bidirectional
        ctx.batch_first = batch_first

        # Fall back to PyTorch for complex cases
        if num_layers > 1 or bidirectional or (dropout > 0 and train):
            return torch.rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)

        return rnn_relu_forward(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)

    @staticmethod
    def backward(ctx, grad_output, grad_hidden):
        logger.debug("GEMS RNN_RELU BACKWARD")

        input, hx = ctx.saved_tensors
        params = ctx.params
        has_biases = ctx.has_biases
        num_layers = ctx.num_layers
        dropout = ctx.dropout
        train = ctx.train
        bidirectional = ctx.bidirectional
        batch_first = ctx.batch_first

        # Fall back to PyTorch for complex cases or gradient computation
        if num_layers > 1 or bidirectional or (dropout > 0 and train):
            # Compute gradients using PyTorch's autograd
            output, hidden = torch.rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)
            grad_input = torch.autograd.grad(output, input, grad_output, retain_graph=True)[0]
            grad_hx = torch.autograd.grad(output, hx, grad_output, retain_graph=True)[0] if hx is not None else None

            # Handle params gradients
            grad_params = []
            for p in params:
                if p.requires_grad:
                    grad_p = torch.autograd.grad(output, p, grad_output, retain_graph=True)[0]
                    grad_params.append(grad_p)
                else:
                    grad_params.append(None)

            # Pad grad_params to match expected length
            while len(grad_params) < len(params):
                grad_params.append(None)

            return grad_input, grad_hx, tuple(grad_params), None, None, None, None, None, None

        # For simple case, use PyTorch's autograd
        output, hidden = rnn_relu_forward(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)

        # Compute gradients
        grad_input = torch.autograd.grad(output, input, grad_output, retain_graph=True)[0]
        grad_hx = torch.autograd.grad(output, hx, grad_output, retain_graph=True)[0] if hx is not None else None

        # Handle params gradients
        grad_params = []
        for p in params:
            if p.requires_grad:
                grad_p = torch.autograd.grad(output, p, grad_output, retain_graph=True)[0]
                grad_params.append(grad_p)
            else:
                grad_params.append(None)

        # Pad grad_params to match expected length
        while len(grad_params) < len(params):
            grad_params.append(None)

        return grad_input, grad_hx, tuple(grad_params), None, None, None, None, None, None


def rnn_relu(input, hx=None, params=None, has_biases=True, num_layers=1, dropout=0.0, train=False, bidirectional=False, batch_first=False):
    """
    Applies an Elman RNN with ReLU activation.

    This is a wrapper that handles parameter management and calls the appropriate implementation.
    """
    logger.debug("GEMS RNN_RELU")

    # If params is None, we need to create it from the input size and hidden size
    if params is None:
        input_size = input.shape[2] if batch_first else input.shape[2]
        hidden_size = 256  # Default, will be inferred

        # Try to infer hidden_size from hx
        if hx is not None:
            hidden_size = hx.shape[2]
        else:
            # Can't proceed without hidden_size
            raise ValueError("Either hx or params must be provided")

        # Create default params (but this won't work well, so fall back to PyTorch)
        logger.debug("GEMS RNN_RELU: No params provided, using PyTorch default")
        return torch.rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)

    # Check if we should use our Triton implementation
    if num_layers == 1 and not bidirectional and dropout == 0:
        return RnnReluFunction.apply(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)
    else:
        # Fall back to PyTorch
        return torch.rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)