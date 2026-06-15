import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

_j0 = tl_extra_shim.j0


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_bessel_j0_func(x):
    return _j0(x.to(tl.float32))


def special_bessel_j0(A):
    logger.debug("GEMS SPECIAL_BESSEL_J0")
    return special_bessel_j0_func(A)


def special_bessel_j0_(A):
    logger.debug("GEMS SPECIAL_BESSEL_J0_")
    special_bessel_j0_func(A, out0=A)
    return A