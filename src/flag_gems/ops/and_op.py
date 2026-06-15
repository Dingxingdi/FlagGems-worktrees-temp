import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def and_func(x, y):
    return x & y


def and_tensor(A, B):
    logger.debug("GEMS AND")
    return and_func(A, B)


def and_tensor_(A, B):
    logger.debug("GEMS AND_")
    return and_func(A, B, out0=A)