import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def resolve_neg_func(x, is_neg):
    # If is_neg is True, negate x, otherwise return x as-is
    return -x if is_neg else x


def resolve_neg(A: torch.Tensor):
    logger.debug("GEMS RESOLVE_NEG")
    is_neg = A.is_neg()
    return resolve_neg_func(A, is_neg)
