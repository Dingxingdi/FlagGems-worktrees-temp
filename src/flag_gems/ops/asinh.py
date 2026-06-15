import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_asinh = tl_extra_shim.asinh

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def asinh_func(x):
    # For float16/bfloat32, compute in float32 for better precision
    x_f32 = x.to(tl.float32)
    return _asinh(x_f32).to(x.dtype)


def asinh(A):
    logger.debug("GEMS ASINH")
    return asinh_func(A)


def asinh_(A):
    logger.debug("GEMS ASINH_")
    asinh_func(A, out0=A)
    return A