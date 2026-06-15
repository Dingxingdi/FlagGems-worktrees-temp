import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def special_erf_func(x):
    output = tl.math.erf(x.to(tl.float32))
    return output


def special_erf(x):
    logger.debug("GEMS special_erf")
    return special_erf_func(x)


def special_erf_(x):
    logger.debug("GEMS special_erf_")
    return special_erf_func(x, out0=x)