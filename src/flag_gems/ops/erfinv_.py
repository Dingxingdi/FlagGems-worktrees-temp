import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

_erfinv = tl_extra_shim.erfinv


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erfinv_func(x):
    return _erfinv(x.to(tl.float32))


def erfinv(x):
    logger.debug("GEMS ERFINV")
    return erfinv_func(x)


def erfinv_(x):
    logger.debug("GEMS ERFINV_")
    erfinv_func(x, out0=x)
    return x