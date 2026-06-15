import logging

import torch

logger = logging.getLogger(__name__)


def fft_rfftn(input, s=None, dim=None, norm=None):
    """N-dimensional discrete Fourier transform of real input.

    This implementation wraps PyTorch's FFT.
    """
    logger.debug("GEMS FFT_RFFTN")
    return torch.fft.rfftn(input, s=s, dim=dim, norm=norm)


def fft_rfftn_(input, s=None, dim=None, norm=None):
    """In-place variant (creates new tensor due to FFT nature)."""
    logger.debug("GEMS FFT_RFFTN_")
    return torch.fft.rfftn(input, s=s, dim=dim, norm=norm)