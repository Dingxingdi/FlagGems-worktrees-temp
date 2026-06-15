import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def special_laguerre_polynomial_l_func(x, n):
    # Compute Laguerre polynomial L_n(x)
    # Using the recursive formula:
    # L_0(x) = 1
    # L_1(x) = 1 - x
    # L_k(x) = ((2k - 1 - x) * L_{k-1}(x) - (k - 1) * L_{k-2}(x)) / k

    # Convert to float32 for computation to avoid type issues
    x32 = x.to(tl.float32)
    n_int = n.to(tl.int32)

    # Base constants
    one = x32 * 0.0 + 1.0
    two = one + 1.0

    # L_0(x) = 1
    l0 = one
    # L_1(x) = 1 - x
    l1 = one - x32

    # Compute L_2 through L_10 iteratively
    # Using recurrence: L_k = ((2k - 1 - x) * L_{k-1} - (k - 1) * L_{k-2}) / k

    kf = tl.constexpr(2.0)
    l2 = ((two * kf - one - x32) * l1 - (kf - one) * l0) / kf

    kf = tl.constexpr(3.0)
    l3 = ((two * kf - one - x32) * l2 - (kf - one) * l1) / kf

    kf = tl.constexpr(4.0)
    l4 = ((two * kf - one - x32) * l3 - (kf - one) * l2) / kf

    kf = tl.constexpr(5.0)
    l5 = ((two * kf - one - x32) * l4 - (kf - one) * l3) / kf

    kf = tl.constexpr(6.0)
    l6 = ((two * kf - one - x32) * l5 - (kf - one) * l4) / kf

    kf = tl.constexpr(7.0)
    l7 = ((two * kf - one - x32) * l6 - (kf - one) * l5) / kf

    kf = tl.constexpr(8.0)
    l8 = ((two * kf - one - x32) * l7 - (kf - one) * l6) / kf

    kf = tl.constexpr(9.0)
    l9 = ((two * kf - one - x32) * l8 - (kf - one) * l7) / kf

    kf = tl.constexpr(10.0)
    l10 = ((two * kf - one - x32) * l9 - (kf - one) * l8) / kf

    # Clamp n to [0, 10] range
    n_clamped = tl.minimum(tl.maximum(n_int, 0), 10)

    # Select the correct result based on n
    # This uses nested tl.where to select from the computed values
    result = n_clamped.to(tl.float32)

    # Use comparison to select - more efficient than nested tl.where
    result = tl.where(result == 0.0, l0,
             tl.where(result == 1.0, l1,
             tl.where(result == 2.0, l2,
             tl.where(result == 3.0, l3,
             tl.where(result == 4.0, l4,
             tl.where(result == 5.0, l5,
             tl.where(result == 6.0, l6,
             tl.where(result == 7.0, l7,
             tl.where(result == 8.0, l8,
             tl.where(result == 9.0, l9, l10))))))))))

    # Convert back to original dtype
    return result.to(x.dtype)


def special_laguerre_polynomial_l(x, n):
    """
    Compute the Laguerre polynomial L_n(x).

    Args:
        x: Tensor of values at which to evaluate the polynomial
        n: Tensor of integer orders (non-negative integers)

    Returns:
        Tensor of L_n(x) values
    """
    logger.debug("GEMS special_laguerre_polynomial_l")
    return special_laguerre_polynomial_l_func(x, n)


def special_laguerre_polynomial_l_(x, n):
    """
    In-place version of special_laguerre_polynomial_l (not applicable for this op).
    For completeness, we implement it but it doesn't modify in-place.
    """
    logger.debug("GEMS special_laguerre_polynomial_l_")
    return special_laguerre_polynomial_l_func(x, n, out0=x)