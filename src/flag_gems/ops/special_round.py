import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def special_round_func(x):
    # Use nearbyint which implements round-half-to-even (banker's rounding)
    # This matches torch.round and torch.special.round behavior
    return tl_extra_shim.nearbyint(x.to(tl.float32)).to(x.dtype)


def special_round(A):
    logger.debug("GEMS SPECIAL_ROUND")
    return special_round_func(A)


def special_round_(A):
    logger.debug("GEMS SPECIAL_ROUND_")
    special_round_func(A, out0=A)
    return A


def special_round_out(A, *, out=None):
    logger.debug("GEMS SPECIAL_ROUND_OUT")
    if out is None:
        return special_round_func(A)
    special_round_func(A, out0=out)
    return out