import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_asin = tl_extra_shim.asin
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def arcsin_kernel(x):
    return _asin(x.to(tl.float32))


def arcsin(A):
    logger.debug("GEMS ARCSIN")
    out = arcsin_kernel(A)
    return out


def arcsin_(A):
    logger.debug("GEMS ARCSIN_")
    arcsin_kernel(A, out0=A)
    return A