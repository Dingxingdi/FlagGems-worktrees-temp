import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_isnan = tl_extra_shim.isnan
_isinf = tl_extra_shim.isinf

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    # Check for NaN
    x_is_nan = _isnan(x.to(tl.float32))
    # Check for positive and negative infinity
    x_is_posinf = _isinf(x.to(tl.float32)) & (x > 0.0)
    x_is_neginf = _isinf(x.to(tl.float32)) & (x < 0.0)
    # Replace values
    x = tl.where(x_is_nan, nan, x)
    x = tl.where(x_is_posinf, posinf, x)
    x = tl.where(x_is_neginf, neginf, x)
    return x


# nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS NAN_TO_NUM")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    return nan_to_num_func(A, nan, posinf, neginf)


# nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)
def nan_to_num_(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS NAN_TO_NUM_")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    nan_to_num_func(A, nan, posinf, neginf, out0=A)
    return A
