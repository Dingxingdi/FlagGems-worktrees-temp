import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fftfftfreq_kernel(
    out_ptr,
    n,
    d,
    step_size,
    half_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n

    # Compute frequency values:
    # f[i] = i / (d * n) for i <= half_n
    # f[i] = (i - n) / (d * n) for i > half_n
    # We use: f[i] = (i - n * (i > half_n)) / (d * n)
    #       = (i - n) / (d * n) for i > half_n
    #       = i / (d * n) for i <= half_n
    offset = idx - n
    # Use where to select: if idx <= half_n use idx, else use (idx - n)
    freq = tl.where(idx <= half_n, idx, offset)
    freq = freq.to(tl.float32)
    freq = freq * step_size

    tl.store(out_ptr + idx, freq, mask=mask)


def fft_fftfreq(n, d=1.0, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS FFTFFTFREQ")

    # Handle dtype: default is torch.float32
    if dtype is None:
        dtype = torch.float32

    # Handle device: default to current device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output tensor
    result = torch.empty((n,), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)

    # Compute step size: 1 / (d * n)
    step_size = 1.0 / (d * n)
    half_n = n // 2

    BLOCK_SIZE = 128
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    fftfftfreq_kernel[grid](
        result,
        n,
        d,
        step_size,
        half_n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return result