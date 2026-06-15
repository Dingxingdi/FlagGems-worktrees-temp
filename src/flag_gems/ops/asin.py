import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_asin = tl_extra_shim.asin
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def asin_kernel(x):
    return _asin(x.to(tl.float32))


def asin(x):
    logger.debug("GEMS ASIN FORWARD")
    y = asin_kernel(x)
    return y


def asin_(x):
    logger.debug("GEMS ASIN")
    asin_kernel(x, out0=x)
    return x