import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def iand_func(x, y):
    return x & y


def __iand__(A, B):
    logger.debug("GEMS __iand__")
    return iand_func(A, B)


def __iand___inplace(A, B):
    logger.debug("GEMS __iand___inplace")
    return iand_func(A, B, out0=A)