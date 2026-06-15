import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_acos = tl_extra_shim.acos
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def arccos_kernel(x):
    # TODO: use flag_gems.utils.tl_extra_shim help apis
    return _acos(x.to(tl.float32))


def arccos(x):
    logger.debug("GEMS ARCCOS FORWARD")
    y = arccos_kernel(x)
    return y


def arccos_(x):
    logger.debug("GEMS ARCCOS_")
    arccos_kernel(x, out0=x)
    return x