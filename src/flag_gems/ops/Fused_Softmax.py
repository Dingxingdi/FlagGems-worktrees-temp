# Implementation of Fused_Softmax - a numerically stable softmax
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def fused_softmax_kernel(
    input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr
):
    """Fused softmax kernel with numerical stability (subtract max before exp)."""
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    row_offset = row_id * n_cols
    x = tl.load(input_ptr + row_offset + cols, mask=mask, other=-float("inf"))
    x_fp32 = x.to(tl.float32)

    # Numerically stable: subtract max before exp
    x_max = tl.max(x_fp32, axis=0)
    all_neginf = x_max == -float("inf")

    x_shifted = x_fp32 - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp

    # Handle all-negative-infinity case (all inputs are -inf)
    softmax = tl.where(all_neginf, tl.zeros([BLOCK_SIZE], dtype=tl.float32), softmax)

    tl.store(output_ptr + row_offset + cols, softmax, mask=mask)


def Fused_Softmax(x: torch.Tensor, dim: int = -1):
    """Numerically stable fused softmax.

    Args:
        x: Input tensor
        dim: Dimension along which to compute softmax (default: -1, last dimension)

    Returns:
        Softmax output tensor
    """
    logger.debug("GEMS FUSED_SOFTMAX")
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.ndim >= 1, "Input tensor must have at least 1 dimension"

    dim = dim if dim >= 0 else x.ndim + dim
    assert 0 <= dim < x.ndim, "Invalid dim for softmax"

    # Handle empty tensor case
    if x.numel() == 0:
        return torch.empty_like(x)

    # If dim is not the last dimension, we need to transpose
    if dim != x.ndim - 1:
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        y = x.permute(perm).contiguous()
        inv_perm = [0] * x.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
    else:
        y = x.contiguous()
        inv_perm = None

    n_cols = y.shape[-1]
    n_rows = y.numel() // n_cols

    # Convert to float32 for computation
    y_fp32 = y.float()
    out_fp32 = torch.empty_like(y_fp32)

    # Determine block size
    def _next_pow2(v: int) -> int:
        if v <= 1:
            return 1
        return 1 << (v - 1).bit_length()

    BLOCK_SIZE = min(4096, _next_pow2(n_cols))
    grid = lambda meta: (n_rows,)

    fused_softmax_kernel[grid](y_fp32, out_fp32, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)

    out = out_fp32.to(x.dtype)

    out = out.view(*y.shape)
    if inv_perm is not None:
        out = out.permute(inv_perm)

    return out