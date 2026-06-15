import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def sign_func(x):
    # Sign function: returns -1 for negative, 0 for zero, 1 for positive
    # Implement manually since tl.sign doesn't exist
    x_fp32 = x.to(tl.float32)
    result = tl.where(x_fp32 > 0, 1.0, tl.where(x_fp32 < 0, -1.0, 0.0))
    return result.to(x.dtype)


def sign(A):
    logger.debug("GEMS SIGN")
    return sign_func(A)


def sign_(A):
    logger.debug("GEMS SIGN_")
    sign_func(A, out0=A)
    return A