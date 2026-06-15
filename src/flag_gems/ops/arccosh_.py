import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_acosh = tl_extra_shim.acosh
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def arccosh_kernel(x):
    # arccosh(x) = ln(x + sqrt(x + 1) * sqrt(x - 1))
    # Use tl_extra_shim.acosh for the computation
    return _acosh(x.to(tl.float32)).to(x.dtype)


def arccosh(A):
    logger.debug("GEMS arccosh")
    return arccosh_kernel(A)


def arccosh_(A):
    logger.debug("GEMS arccosh_")
    arccosh_kernel(A, out0=A)
    return A