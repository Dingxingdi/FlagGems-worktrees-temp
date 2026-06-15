import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_asin = tl_extra_shim.asin
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def arcsin_kernel(x):
    return _asin(x.to(tl.float32))


def arcsin(x):
    logger.debug("GEMS ARCSIN FORWARD")
    y = arcsin_kernel(x)
    return y


def arcsin_(x):
    logger.debug("GEMS ARCSIN INPLACE")
    arcsin_kernel(x, out0=x)
    return x


def arcsin_out(x, *, out=None):
    logger.debug("GEMS ARCSIN OUT")
    if out is None:
        return arcsin_kernel(x)
    arcsin_kernel(x, out0=out)
    return out