import logging

import torch

# Get reference to original torch.fft.hfft function BEFORE flag_gems is fully loaded
# This captures the original implementation
_original_fft_hfft = torch.fft.hfft

import flag_gems
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)


def fft_hfft(A: torch.Tensor, n: int = None, dim: int = -1, norm: str = None):
    """
    Compute the inverse FFT from a half-Hermitian spectrum.

    This implements hfft which takes a half-Hermitian complex signal and
    returns a real-valued signal.

    Args:
        A: Input complex tensor representing half-Hermitian signal
        n: Output signal length (optional)
        dim: Dimension along which to compute FFT (default: -1)
        norm: Normalization mode ("forward", "backward", "ortho", or None)

    Returns:
        Real-valued tensor with the full FFT result
    """
    logger.debug("GEMS FFT_HFFT")

    # Call the original torch.fft.hfft function that we captured at module load time
    # This bypasses the FlagGems dispatcher
    return _original_fft_hfft(A, n=n, dim=dim, norm=norm)


def fft_hfft_(A: torch.Tensor, n: int = None, dim: int = -1, norm: str = None):
    """
    In-place version of fft_hfft (not supported).
    FFT operations don't support in-place computation.
    """
    raise NotImplementedError("fft_hfft_ in-place is not supported")