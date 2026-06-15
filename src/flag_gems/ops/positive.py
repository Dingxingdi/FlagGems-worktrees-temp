import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def positive_func(x):
    return x


def positive(A):
    logger.debug("GEMS POSITIVE")
    return positive_func(A)


def positive_(A):
    logger.debug("GEMS POSITIVE_")
    positive_func(A, out0=A)
    return A