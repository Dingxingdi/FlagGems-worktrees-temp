import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def not_func(x):
    return not x.to(tl.int1)


def not_(A):
    logger.debug("GEMS NOT")
    return not_func(A)