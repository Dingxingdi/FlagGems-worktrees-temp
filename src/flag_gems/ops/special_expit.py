import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_expit_forward(x):
    # log2e: tl.constexpr = math.log2(math.e)
    # triton 3.0.0 disallow calling non-jitted function inside jitted function, even if it is in
    # the rhs of an assignment to a constexpr, so we use numeric literal instead to work around this.
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + exp2(-x.to(tl.float32) * log2e))


def special_expit(A):
    logger.debug("GEMS special_expit FORWARD")
    output = special_expit_forward(A)
    return output


def special_expit_(A):
    logger.debug("GEMS special_expit_ FORWARD")
    out = special_expit_forward(A, out0=A)
    return out