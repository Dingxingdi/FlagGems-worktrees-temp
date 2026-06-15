import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def range_func(y_ptr, start, end, size, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    y_ptr += pid * BLOCK_SIZE
    offset = pid * BLOCK_SIZE

    cols = tl.arange(0, BLOCK_SIZE)
    range_val = cols + offset + start
    mask = cols + pid * BLOCK_SIZE
    tl.store(y_ptr + cols, range_val, mask=mask < size)


import sys

def range_op(
    start, end, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS RANGE")
    start = float(start)
    end = float(end)

    if dtype is None:
        # If any of start/end are float, use float64, otherwise int64
        if isinstance(start, float) or isinstance(end, float):
            dtype = torch.float64
        else:
            dtype = torch.int64

    if dtype in (torch.int32, torch.int64):
        start = int(start)
        end = int(end)
        size = end - start + 1
    else:
        # Float types
        size = math.ceil(end - start) + 1

    size = int(size)
    if size < 0:
        size = 0

    BLOCK_SIZE = 128
    grid = triton.cdiv(size, BLOCK_SIZE)

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = runtime.device.name

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    range_func[grid,](result, start, end, size, BLOCK_SIZE)
    return result