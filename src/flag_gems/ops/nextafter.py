import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

# Get the libdevice nextafter function
_nextafter = tl_extra_shim.nextafter


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def nextafter_func(input, other):
    # IEEE 754 nextafter: returns the next representable float after input towards other
    # Use libdevice's nextafter which supports float32/float64
    # For float16/bfloat16, convert to float32 first
    dtype = input.dtype
    if dtype.is_fp16():
        x = input.to(tl.float32)
        y = other.to(tl.float32)
        result = _nextafter(x, y)
        return result.to(tl.float16)
    elif dtype.is_bf16():
        x = input.to(tl.float32)
        y = other.to(tl.float32)
        result = _nextafter(x, y)
        return result.to(tl.bfloat16)
    else:
        # float32 or float64
        return _nextafter(input, other)


def nextafter(input, other, *, out=None):
    logger.debug("GEMS NEXTAFTER")
    return nextafter_func(input, other)


def nextafter_(input, other, *, out=None):
    logger.debug("GEMS NEXTAFTER_")
    if out is None:
        return nextafter_func(input, other, out0=input)
    nextafter_func(input, other, out0=out)
    return out