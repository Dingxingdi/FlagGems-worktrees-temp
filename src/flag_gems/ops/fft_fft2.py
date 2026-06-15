import logging

import torch

logger = logging.getLogger(__name__)


def fft_fft2(inp, s=None, dim=(-2, -1), norm=None):
    """
    2D FFT operation.

    Args:
        inp: Input tensor (complex64 or complex128)
        s: Output shape (optional)
        dim: Dimensions to apply FFT (default: (-2, -1))
        norm: Normalization mode (None, "forward", "backward", "ortho")

    Returns:
        Complex tensor with 2D FFT applied
    """
    logger.debug("GEMS fft_fft2")

    if not inp.is_complex():
        raise ValueError("fft_fft2 requires complex input tensor")

    # Use torch._refs.fft.fft2 which bypasses the FlagGems dispatcher
    return torch._refs.fft.fft2(inp, s=s, dim=dim, norm=norm)