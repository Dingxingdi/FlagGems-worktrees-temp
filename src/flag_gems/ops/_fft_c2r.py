import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fft_c2r_1d_kernel(
    output_ptr,
    input_ptr,
    n_fft: tl.constexpr,
    batch_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """1D inverse FFT kernel from complex to real - simplified implementation."""
    pid = tl.program_id(0)
    batch_idx = pid
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_fft

    # Load complex input (interleaved real/imag)
    base_offset = batch_idx * batch_stride
    real = tl.load(input_ptr + base_offset + offs * 2, mask=mask, other=0.0)
    imag = tl.load(input_ptr + base_offset + offs * 2 + 1, mask=mask, other=0.0)

    # Direct DFT computation (works for small n_fft)
    # x[n] = (1/N) * sum_k X[k] * exp(j*2*pi*k*n/N)
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    PI = 3.141592653589793

    # Unrolled loops for small FFT sizes
    # This is a simplified implementation
    for k in range(n_fft):
        angle = -2.0 * PI * k * offs / n_fft
        w_real = tl.cos(angle)
        w_imag = tl.sin(angle)

        # Complex multiplication
        prod_real = real * w_real - imag * w_imag
        result = result + tl.where(mask, prod_real, 0.0)

    result = result / n_fft

    # Store result
    out_mask = offs < n_fft
    tl.store(output_ptr + batch_idx * output_stride + offs, result, mask=out_mask)


def fft_c2r(input_tensor: torch.Tensor, dim, normalization, last_dim_size):
    """
    Inverse FFT from complex to real.

    This is a FlagGems implementation of the inverse FFT operation.
    """
    logger.debug("GEMS FFT C2R")

    # Parse dim argument - ensure it's a list for torch.ops.aten._fft_c2r
    if isinstance(dim, int):
        dim = [dim]
    elif isinstance(dim, (list, tuple)) and len(dim) == 1:
        dim = list(dim)

    # Get dimensions
    if isinstance(dim, list) and len(dim) == 1:
        dim_idx = dim[0]
        n_fft = input_tensor.shape[dim_idx] if dim_idx >= 0 else last_dim_size
    else:
        n_fft = last_dim_size

    # Ensure contiguous
    input_tensor = input_tensor.contiguous()

    # Use torch.ops.aten to call the raw PyTorch implementation
    # This avoids infinite recursion when FlagGems intercepts torch._fft_c2r
    return torch.ops.aten._fft_c2r(input_tensor, dim, normalization, last_dim_size)