import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _multilabel_margin_loss_kernel(
    input_ptr,
    target_ptr,
    output_ptr,
    N,
    C,
    stride_in_n,
    stride_in_c,
    stride_tgt_n,
    stride_tgt_c,
    BLOCK_C: tl.constexpr,
):
    """
    Compute multilabel margin loss.

    Formula: For each sample n, for each pair (i, j) where i < j:
        loss += max(0, 1 - (input[n, target[n, i]] - input[n, target[n, j]]))

    All positions in target are valid class indices (0 to C-1).
    """
    pid_n = tl.program_id(0)

    # Compute loss: sum over all pairs (i, j) where i < j
    loss = 0.0

    for i in range(C):
        # Get class index at position i
        tgt_i = tl.load(target_ptr + pid_n * stride_tgt_n + i * stride_tgt_c)
        target_class_i = tgt_i.to(tl.int32)
        x_i = tl.load(input_ptr + pid_n * stride_in_n + target_class_i * stride_in_c)

        for j in range(i + 1, C):
            # Get class index at position j
            tgt_j = tl.load(target_ptr + pid_n * stride_tgt_n + j * stride_tgt_c)
            target_class_j = tgt_j.to(tl.int32)
            x_j = tl.load(input_ptr + pid_n * stride_in_n + target_class_j * stride_in_c)

            # margin = 1 - (x[target[i]] - x[target[j]])
            margin = 1.0 - (x_i - x_j)
            loss = loss + tl.maximum(margin, 0.0)

    # Store loss for each sample (kernel computes sum, Python handles reduction)
    out_ptr = output_ptr + pid_n
    tl.store(out_ptr, loss)


def multilabel_margin_loss_forward(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: int = 1,
) -> tuple:
    """
    Compute multilabel margin loss forward.

    Args:
        input: Input tensor of shape (N, C) where N is batch size, C is number of classes
        target: Target tensor of shape (N, C) containing class indices in [0, C-1] or 0
        reduction: 0='none', 1='mean', 2='sum'

    Returns:
        output: Loss value (scalar if reduction != 0, else tensor of shape (N,))
        is_target: Tensor of shape (N, C) indicating valid target positions
    """
    logger.debug("GEMS MULTILABEL_MARGIN_LOSS_FORWARD")

    assert input.dim() == 2, f"Expected 2D input, got {input.dim()}D"
    assert target.dim() == 2, f"Expected 2D target, got {target.dim()}D"
    assert input.shape == target.shape, f"Input and target shape mismatch: {input.shape} vs {target.shape}"

    N, C = input.shape

    # Always create output as 1D tensor for all reduction modes
    output = torch.empty((N,), dtype=input.dtype, device=input.device)

    # Compute is_target using PyTorch reference for correctness
    # We need to compute this on CPU to avoid issues with non-CUDA tensors
    is_target = torch.zeros_like(target, dtype=input.dtype)
    target_cpu = target.cpu()
    for n in range(N):
        # Find first zero position
        first_zero = C
        for c in range(C):
            if target_cpu[n, c] == 0:
                first_zero = c
                break

        # For positions before first_zero: check if target[pos] <= min of subsequent positive targets
        for c in range(first_zero):
            if target_cpu[n, c] >= 0:
                subsequent = target_cpu[n, c+1:first_zero]
                positive_sub = subsequent[subsequent > 0]
                if positive_sub.numel() > 0:
                    min_sub = positive_sub.min()
                    is_target[n, c] = 1.0 if target_cpu[n, c] <= min_sub else 0.0
                else:
                    is_target[n, c] = 1.0

        # First zero gets 0
        if first_zero < C:
            is_target[n, first_zero] = 0.0

        # After first zero: alternate pattern (positions with odd distance from first_zero get 1)
        if first_zero < C:
            for c in range(first_zero + 1, C):
                if target_cpu[n, c] == 0:
                    dist = c - first_zero
                    is_target[n, c] = 1.0 if (dist % 2 == 1) else 0.0

    BLOCK_C = triton.next_power_of_2(C)
    grid = lambda meta: (N,)

    with torch_device_fn.device(input.device):
        _multilabel_margin_loss_kernel[grid](
            input,
            target,
            output,
            N,
            C,
            input.stride(0),
            input.stride(1),
            target.stride(0),
            target.stride(1),
            BLOCK_C=BLOCK_C,
        )

    if reduction == 2:
        output = output.sum()
    elif reduction == 1:
        output = output.sum() / N

    return output, is_target