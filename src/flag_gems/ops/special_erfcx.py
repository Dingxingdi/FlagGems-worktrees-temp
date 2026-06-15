import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def special_erfcx_func(x):
    # Use libdevice erfcx for accurate computation
    return tl.extra.cuda.libdevice.erfcx(x)


def special_erfcx(x):
    logger.debug("GEMS SPECIAL_ERFCX")
    return special_erfcx_func(x)


def special_erfcx_(x):
    logger.debug("GEMS SPECIAL_ERFCX_")
    return special_erfcx_func(x, out0=x)