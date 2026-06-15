import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def gcd_func(x, y):
    # Euclidean algorithm for GCD
    a = tl.abs(x)
    b = tl.abs(y)

    # Run enough iterations for all integer types
    # 32 iterations is enough for any int32 values
    for i in tl.static_range(32):
        # Compute r = a % b only when b != 0
        # When b == 0, we use b = 1 to avoid division by zero
        r = a % tl.where(b != 0, b, 1)
        # Update: a = b, b = r
        a = tl.where(b != 0, b, a)
        b = r

    # When loop exits, a contains the GCD
    return a


def gcd(A, B):
    logger.debug("GEMS GCD")
    return gcd_func(A, B)


def gcd_(A, B):
    logger.debug("GEMS GCD_")
    return gcd_func(A, B, out0=A)