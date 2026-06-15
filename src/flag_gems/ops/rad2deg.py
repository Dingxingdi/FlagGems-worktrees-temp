import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

RAD2DEG_FACTOR: tl.constexpr = 180.0 / 3.141592653589793


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def rad2deg_func(x):
    return (x.to(tl.float32) * RAD2DEG_FACTOR).to(x.dtype)


def rad2deg(A):
    logger.debug("GEMS RAD2DEG")
    return rad2deg_func(A)


def rad2deg_(A):
    logger.debug("GEMS RAD2DEG_")
    rad2deg_func(A, out0=A)
    return A