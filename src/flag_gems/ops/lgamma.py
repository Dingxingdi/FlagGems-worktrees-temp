import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_lgamma = tl_extra_shim.lgamma

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def lgamma_func(x):
    return _lgamma(x.to(tl.float32)).to(x.dtype)


def lgamma(A):
    logger.debug("GEMS LGAMMA")
    return lgamma_func(A)


def lgamma_(A):
    logger.debug("GEMS LGAMMA_")
    lgamma_func(A, out0=A)
    return A