import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def expm1_func(x):
    # expm1(x) = exp(x) - 1
    # Use float32 for computation then convert back
    return (tl.exp(x.to(tl.float32)) - 1.0).to(x.dtype)


def expm1(A):
    logger.debug("GEMS EXPM1")
    return expm1_func(A)


def expm1_(A):
    logger.debug("GEMS EXPM1_")
    expm1_func(A, out0=A)
    return A