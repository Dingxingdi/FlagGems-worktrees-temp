import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def less_func(x, y):
    return x.to(tl.float32) < y.to(tl.float32)


def less(A, B):
    logger.debug("GEMS LESS")
    return less_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def less_func_scalar(x, y):
    return x.to(tl.float32) < y


def less_scalar(A, B):
    logger.debug("GEMS LESS SCALAR")
    return less_func_scalar(A, B)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def less_func_scalar_rev(x, y):
    return x.to(tl.float32) < y.to(tl.float32)


def less_scalar_rev(A, B):
    logger.debug("GEMS LESS SCALAR REV")
    return less_func_scalar_rev(A, B)


# Alias for lt (torch.lt / torch.less)
lt = less
lt_scalar = less_scalar
lt_scalar_rev = less_scalar_rev