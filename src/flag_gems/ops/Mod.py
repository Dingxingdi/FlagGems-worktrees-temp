import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _mod(x, y):
    # Python's modulus operation:
    # The result has the same sign as the divisor (y) and its absolute value is less than that of y.
    # Implementation: r = x % y, then adjust if signs differ and r != 0
    r = x % y
    c1 = r != 0
    c2 = (x < 0) ^ (y < 0)
    return tl.where(c1 & c2, r + y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mod_tt(x, y):
    return _mod(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mod_ts(x, y):
    return _mod(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mod_st(x, y):
    return _mod(x, y)


def Mod(A, B):
    """Mod operator - computes the remainder element-wise.

    The result has the same sign as the divisor (other) and its absolute value
    is less than that of other.

    This is equivalent to Python's % operator and torch.remainder.
    """
    logger.debug("GEMS MOD")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return mod_tt(A, B)
    elif isinstance(A, torch.Tensor):
        return mod_ts(A, B)
    elif isinstance(B, torch.Tensor):
        return mod_st(A, B)
    else:
        # Both scalar
        return torch.tensor(A % B)


def Mod_(A, B):
    """In-place Mod operator."""
    logger.debug("GEMS MOD_")
    if isinstance(B, torch.Tensor):
        return mod_tt(A, B, out0=A)
    else:
        return mod_ts(A, B, out0=A)