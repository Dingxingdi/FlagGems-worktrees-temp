import logging

import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def sinc_func(x):
    # Compute normalized sinc: sin(pi * x) / (pi * x)
    # When x == 0, sinc(x) = 1
    x_f32 = x.to(tl.float32)
    pi = 3.141592653589793
    # Handle x == 0 case to avoid division by zero
    # sinc(0) = 1
    return tl.where(x_f32 == 0.0, 1.0, tl.sin(pi * x_f32) / (pi * x_f32))


def sinc(A):
    logger.debug("GEMS SINC")
    return sinc_func(A)


def sinc_(A):
    logger.debug("GEMS SINC_")
    sinc_func(A, out0=A)
    return A