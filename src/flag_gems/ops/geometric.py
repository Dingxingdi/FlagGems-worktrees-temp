import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def geometric_func(x, p):
    # Geometric distribution sampling using inverse CDF method
    # For uniform u in (0, 1), geometric distribution with success probability p:
    # k = floor(log(1-u) / log(1-p)) + 1
    one_minus_x = 1.0 - x
    one_minus_p = 1.0 - p
    return tl.floor(tl.log(one_minus_x.to(tl.float32)) / tl.log(one_minus_p)) + 1.0


def geometric(A, p):
    logger.debug("GEMS GEOMETRIC")
    return geometric_func(A, p)


def geometric_(A, p):
    logger.debug("GEMS GEOMETRIC_")
    geometric_func(A, p, out0=A)
    return A