import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erfc_func(x):
    output = 1.0 - tl.math.erf(x.to(tl.float32))
    return output


def erfc(x):
    logger.debug("GEMS ERFC")
    return erfc_func(x)


def erfc_(x):
    logger.debug("GEMS ERFC_")
    return erfc_func(x, out0=x)