import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_asin = tl_extra_shim.asin
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def asin_kernel(x):
    # TODO: use flag_gems.utils.tl_extra_shim help apis
    return _asin(x.to(tl.float32))


def asin(A):
    logger.debug("GEMS ASIN FORWARD")
    return asin_kernel(A)


def asin_(A):
    logger.debug("GEMS ASIN_")
    asin_kernel(A, out0=A)
    return A