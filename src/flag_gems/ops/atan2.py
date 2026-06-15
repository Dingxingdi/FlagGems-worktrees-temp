import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_atan2 = tl_extra_shim.atan2

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def atan2_func(y, x):
    return _atan2(y.to(tl.float32), x.to(tl.float32)).to(y.dtype)


def atan2(A, B):
    logger.debug("GEMS ATAN2")
    return atan2_func(A, B)


def atan2_(A, B):
    logger.debug("GEMS ATAN2_")
    atan2_func(A, B, out0=A)
    return A
