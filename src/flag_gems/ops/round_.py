import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def round_func(x):
    # Use tl_extra_shim.round which implements "round half to even"
    x_fp32 = x.to(tl.float32)
    return tl_extra_shim.nearbyint(x_fp32).to(x.dtype)


def round(A):
    logger.debug("GEMS ROUND")
    return round_func(A)


def round_(A):
    logger.debug("GEMS ROUND_")
    round_func(A, out0=A)
    return A