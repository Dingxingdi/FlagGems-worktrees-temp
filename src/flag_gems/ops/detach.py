import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def detach_func(x):
    return x


def detach(A):
    logger.debug("GEMS DETACH")
    return detach_func(A)


def detach_(A):
    logger.debug("GEMS DETACH_")
    detach_func(A, out0=A)
    return A