import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def lgamma_func(x):
    return tl_extra_shim.lgamma(x.to(tl.float32))


def lgamma(A):
    logger.debug("GEMS LGAMMA")
    return lgamma_func(A)


def lgamma_(A):
    logger.debug("GEMS LGAMMA_")
    lgamma_func(A, out0=A)
    return A


def lgamma_out(A, out):
    logger.debug("GEMS LGAMMA_OUT")
    if out is None:
        return lgamma_func(A)
    lgamma_func(A, out0=out)
    return out