import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def log10_func(x):
    # log10(x) = log(x) / log(10) = log(x) * (1 / log(10))
    # 1 / ln(10) ≈ 0.4342944819032518
    return (tl.log(x.to(tl.float32)) * 0.4342944819032518).to(x.dtype)


def log10_(A):
    logger.debug("GEMS LOG10_")
    log10_func(A, out0=A)
    return A