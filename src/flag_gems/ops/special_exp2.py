import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_exp2_func(x):
    return tl.exp2(x.to(tl.float32))


def special_exp2(A):
    logger.debug("GEMS SPECIAL_EXP2")
    return special_exp2_func(A)