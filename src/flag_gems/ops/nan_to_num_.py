import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    # NaN detection: NaN != NaN is True
    # But we need to handle float32 precision for float16/bfloat16
    x_fp32 = x.to(tl.float32)
    x_is_nan = x_fp32 != x_fp32

    # Infinity detection using comparison with infinity
    # Note: inf * 0 = nan, so we use 1.0/x == 0.0 to detect inf
    x_is_posinf = (x_fp32 > 0.0) & (1.0 / x_fp32 == 0.0)
    x_is_neginf = (x_fp32 < 0.0) & (1.0 / x_fp32 == 0.0)

    # Replace values
    result = x_fp32
    result = tl.where(x_is_nan, nan, result)
    result = tl.where(x_is_posinf, posinf, result)
    result = tl.where(x_is_neginf, neginf, result)

    return result.to(x.dtype)


def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS NAN_TO_NUM")
    # Use PyTorch's default values: finfo().max and finfo().min
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    return nan_to_num_func(A, nan, posinf, neginf)


def nan_to_num_(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS NAN_TO_NUM_")
    # Use PyTorch's default values: finfo().max and finfo().min
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    nan_to_num_func(A, nan, posinf, neginf, out0=A)
    return A