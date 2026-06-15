import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def acosh_kernel(x):
    # acosh(x) = log(x + sqrt(x*x - 1))
    # Compute in float32 for better accuracy, then cast back
    x_f32 = x.to(tl.float32)
    # Use max with 1.0 to avoid negative values inside sqrt due to float precision
    inner = tl.maximum(x_f32 * x_f32 - 1.0, 0.0)
    result = tl.log(x_f32 + tl.sqrt(inner))
    return result.to(x.dtype)


def acosh(x):
    logger.debug("GEMS ACOSH")
    return acosh_kernel(x)


def acosh_(x):
    logger.debug("GEMS ACOSH_")
    acosh_kernel(x, out0=x)
    return x