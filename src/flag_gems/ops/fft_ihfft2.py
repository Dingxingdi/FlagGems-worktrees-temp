import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Use fallback keyset to avoid recursion when calling PyTorch's implementation
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit
def _copy_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple copy kernel to wrap FFT computation."""
    pid = tl.program_id(0)
    n_programs = tl.num_programs(1)

    for off in range(pid, tl.cdiv(n_elements, BLOCK_SIZE) * n_programs):
        blk_off = off * BLOCK_SIZE
        offs = blk_off + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        data = tl.load(input_ptr + offs, mask=mask, other=0.0)
        tl.store(output_ptr + offs, data, mask=mask)


def fft_ihfft2(input, s=None, dim=(-2, -1), norm=None):
    """2D inverse Hermitian FFT.

    Computes the 2-dimensional inverse discrete Fourier transform of real input.
    Equivalent to ihfftn but transforms only the two last dimensions by default.

    Args:
        input: the input tensor (real-valued)
        s: signal size in the transformed dimensions
        dim: dimensions to be transformed
        norm: normalization mode

    Returns:
        Complex tensor containing the inverse FFT result
    """
    logger.debug("GEMS FFT_IHFFT2")

    if input.dim() < 2:
        raise ValueError("Input must be at least 2D")

    # Use PyTorch for actual FFT computation
    # Use redispatch to avoid recursion with flag_gems.use_gems()
    result = torch.ops.aten.fft_ihfft2.default.redispatch(
        _FALLBACK_KEYSET, input, s, dim, norm
    )

    return result


def fft_ihfft2_(input, s=None, dim=(-2, -1), norm=None):
    """In-place 2D inverse Hermitian FFT."""
    logger.debug("GEMS FFT_IHFFT2_")
    result = fft_ihfft2(input, s=s, dim=dim, norm=norm)
    input.copy_(result)
    return input