import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sinc_func(x):
    # sinc(x) = sin(pi * x) / (pi * x), with sinc(0) = 1 (by continuity)
    pi = 3.141592653589793
    px = pi * x.to(tl.float32)
    # Handle x == 0 case to avoid 0/0, sinc(0) = 1
    return tl.where(x == 0.0, 1.0, tl.sin(px) / px)


def sinc(A):
    logger.debug("GEMS SINC")
    return sinc_func(A)


def sinc_(A):
    logger.debug("GEMS SINC_")
    sinc_func(A, out0=A)
    return A