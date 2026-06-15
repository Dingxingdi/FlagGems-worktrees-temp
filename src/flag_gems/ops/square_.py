import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def square_func(x):
    return (x.to(tl.float32) * x.to(tl.float32)).to(x.dtype)


def square(A):
    logger.debug("GEMS SQUARE")
    return square_func(A)


def square_(A):
    logger.debug("GEMS SQUARE_")
    square_func(A, out0=A)
    return A