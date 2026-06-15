import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_erfinv = tl_extra_shim.erfinv

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def ndtri_func(x):
    # ndtri(p) = sqrt(2) * erfinv(2p - 1)
    SQRT2 = 1.4142135623730951
    return SQRT2 * _erfinv((2.0 * x.to(tl.float32)) - 1.0)


def special_ndtri(A):
    logger.debug("GEMS SPECIAL_NDTRI")
    return ndtri_func(A)


def special_ndtri_(A):
    logger.debug("GEMS SPECIAL_NDTRI_")
    ndtri_func(A, out0=A)
    return A