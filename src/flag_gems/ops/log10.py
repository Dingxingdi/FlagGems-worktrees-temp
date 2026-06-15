import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def log10_func(x):
    # log10(x) = log(x) / log(10)
    return (tl.log(x.to(tl.float32)) / tl.log(tl.constexpr(10.0))).to(x.dtype)


def log10(A):
    logger.debug("GEMS LOG10")
    return log10_func(A)


def log10_(A):
    logger.debug("GEMS LOG10_")
    log10_func(A, out0=A)
    return A