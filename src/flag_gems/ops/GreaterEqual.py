import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# Reuse the ge (greater or equal) implementation since greater_equal is functionally identical
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater_equal_func(x, y):
    return x.to(tl.float32) >= y


def greater_equal(A, B):
    if A.device != B.device:
        device = flag_gems.runtime.device.name
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS GREATER_EQUAL")
    return greater_equal_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def greater_equal_func_scalar(x, y):
    return x.to(tl.float32) >= y


def greater_equal_scalar(A, B):
    logger.debug("GEMS GREATER_EQUAL SCALAR")
    return greater_equal_func_scalar(A, B)