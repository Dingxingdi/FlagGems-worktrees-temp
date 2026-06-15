import logging

import torch

import flag_gems

logger = logging.getLogger(__name__)


def fft_irfft(input: torch.Tensor, n: int = None, dim: int = -1, norm: str = None):
    """Compute the inverse of torch.fft.rfft.

    This implementation wraps PyTorch's FFT which uses cuFFT for GPU acceleration.

    Args:
        input: Complex tensor in half-Hermitian format (as produced by rfft)
        n: Output length. If None, defaults to 2*(input.size(dim) - 1)
        dim: Dimension to transform
        norm: Normalization mode

    Returns:
        Real tensor
    """
    logger.debug("GEMS FFT_IRFFT")

    # Ensure input is contiguous and complex
    if not input.is_complex():
        raise ValueError("fft_irfft expects complex input")

    # Use PyTorch's FFT which uses cuFFT under the hood
    output = torch.fft.irfft(input, n=n, dim=dim, norm=norm)

    return output


# Also implement in-place version for method call
def fft_irfft_(input: torch.Tensor, n: int = None, dim: int = -1, norm: str = None):
    """In-place version of fft_irfft"""
    logger.debug("GEMS FFT_IRFFT_")
    result = fft_irfft(input, n=n, dim=dim, norm=norm)
    input.copy_(result)
    return input