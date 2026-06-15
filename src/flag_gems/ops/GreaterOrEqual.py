import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater_or_equal_func(x, y):
    return x.to(tl.float32) >= y.to(tl.float32)


def GreaterOrEqual(A, B):
    if A.device != B.device:
        if A.device.type == "cuda":
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS GreaterOrEqual")
    return greater_or_equal_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater_or_equal_func_scalar(x, y):
    return x.to(tl.float32) >= y.to(tl.float32)


def GreaterOrEqual_scalar(A, B):
    logger.debug("GEMS GreaterOrEqual SCALAR")
    return greater_or_equal_func_scalar(A, B)