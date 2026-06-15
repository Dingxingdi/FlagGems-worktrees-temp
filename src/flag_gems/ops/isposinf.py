import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_isinf = tl_extra_shim.isinf

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isposinf_func(x):
    # Convert to float32 for consistent isinf behavior, then check if positive infinity
    x_fp32 = x.to(tl.float32)
    return _isinf(x_fp32) & (x_fp32 > 0)


def isposinf(A):
    logger.debug("GEMS ISPOSINF")
    return isposinf_func(A)