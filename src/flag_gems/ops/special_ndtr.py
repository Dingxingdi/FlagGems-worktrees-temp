import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def ndtr_func(x):
    # ndtr(x) = 0.5 * (1 + erf(x / sqrt(2)))
    # Compute in float32 for precision, then cast back
    x_f32 = x.to(tl.float32)
    SQRT_2 = 1.4142135623730951  # sqrt(2)
    result = 0.5 * (1.0 + tl.math.erf(x_f32 / SQRT_2))
    return result.to(x.dtype)


def special_ndtr(A):
    logger.debug("GEMS SPECIAL_NDTR")
    return ndtr_func(A)


def special_ndtr_(A):
    logger.debug("GEMS SPECIAL_NDTR_")
    ndtr_func(A, out0=A)
    return A