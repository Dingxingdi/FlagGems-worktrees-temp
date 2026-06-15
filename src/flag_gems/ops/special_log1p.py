import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def special_log1p_func(x):
    # log1p(x) = log(x + 1), computed in float32 for precision
    return tl.log(x.to(tl.float32) + 1.0).to(x.dtype)


def special_log1p(A):
    logger.debug("GEMS SPECIAL_LOG1P")
    return special_log1p_func(A)


# Also provide log1p as an alias (torch.special.log1p is alias for torch.log1p)
def log1p(A):
    logger.debug("GEMS LOG1P")
    return special_log1p_func(A)


def special_log1p_(A):
    logger.debug("GEMS SPECIAL_LOG1P_")
    special_log1p_func(A, out0=A)
    return A


# Alias for in-place version
def log1p_(A):
    logger.debug("GEMS LOG1P_")
    special_log1p_func(A, out0=A)
    return A