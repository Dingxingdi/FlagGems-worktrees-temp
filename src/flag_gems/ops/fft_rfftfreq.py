import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def rfftfreq_kernel(
    out_ptr,
    n,
    d,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < size

    # out[idx] = idx / (d * n)
    # We compute as idx / (d * n) = (idx / n) / d
    # But to avoid floating point issues, we compute idx / (d * n) directly
    k = idx.to(tl.float32)
    n_f = n.to(tl.float32)
    d_f = d.to(tl.float32)

    val = k / (d_f * n_f)
    tl.store(out_ptr + idx, val, mask=mask)


def fft_rfftfreq(n, d=1.0, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS FFT_RFFTFREQ")
    if isinstance(n, torch.Tensor):
        n = n.item()
    if isinstance(d, torch.Tensor):
        d = d.item()

    n = int(n)
    d = float(d)

    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")

    size = n // 2 + 1

    if dtype is None:
        dtype = torch.float32

    if layout is None:
        layout = torch.strided

    if device is None:
        device = torch.cuda.current_device()

    if pin_memory is None:
        pin_memory = False

    result = torch.empty(
        (size,),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )

    BLOCK_SIZE = 128
    grid = (triton.cdiv(size, BLOCK_SIZE),)

    rfftfreq_kernel[grid](
        result,
        n,
        d,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return result