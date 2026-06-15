import logging

import torch

logger = logging.getLogger(__name__)


def istft(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int = None,
    win_length: int = None,
    window: torch.Tensor = None,
    center: bool = True,
    normalized: bool = False,
    onesided: bool = None,
    length: int = None,
    return_complex: bool = False,
):
    """
    Inverse Short-Time Fourier Transform (ISTFT).

    This implementation delegates to PyTorch's native istft.
    """
    logger.debug("GEMS ISTFT")

    result = torch.istft(
        input,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        length=length,
        return_complex=return_complex,
    )

    return result