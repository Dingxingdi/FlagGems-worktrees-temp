import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def gt__func(x, y):
    return x.to(tl.float32) > y


def gt_(A, B):
    logger.debug("GEMS GT_")
    if isinstance(B, torch.Tensor):
        return gt__func(A, B, out0=A)
    else:
        return gt__func_scalar(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def gt__func_scalar(x, y):
    return x.to(tl.float32) > y


def gt__scalar(A, B):
    logger.debug("GEMS GT_ SCALAR")
    return gt__func_scalar(A, B, out0=A)