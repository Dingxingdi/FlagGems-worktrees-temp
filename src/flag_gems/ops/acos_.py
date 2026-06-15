import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_acos = tl_extra_shim.acos

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def acos_func(x):
    return _acos(x.to(tl.float32))


def acos(A):
    logger.debug("GEMS ACOS")
    return acos_func(A)


def acos_(A):
    logger.debug("GEMS ACOS_")
    acos_func(A, out0=A)
    return A