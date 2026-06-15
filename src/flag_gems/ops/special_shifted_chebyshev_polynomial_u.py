import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _shifted_chebyshev_u_func(x, n):
    # Shifted Chebyshev polynomial of second kind: U_n*(x) = U_n(2x - 1)
    # Recurrence: U_0*=1, U_1*=2*(2x-1), U_n*=2*(2x-1)*U_{n-1}* - U_{n-2}*
    # Compute in fp32 for numerical stability, then cast back to original dtype
    xf = x.to(tl.float32)
    y = 2 * xf - 1

    # Cast n to int32 for comparison
    n_i32 = n.to(tl.int32)

    # Compute using recurrence relation
    # U_0 = 1, U_1 = 2*y
    # Use result to accumulate the final value based on n
    u0 = tl.cast(1.0, tl.float32)
    u1 = tl.cast(2.0, tl.float32) * y

    # Compute iteratively but maintain type consistency
    result = u0
    result = tl.where(n_i32 == 0, u0, result)
    result = tl.where(n_i32 == 1, u1, result)

    # For higher n, compute iteratively
    u_km2 = u0
    u_km1 = u1

    # Manually unroll a few iterations for common small n values
    # Then use a loop for the rest
    # k=2: U_2 = 2*y*U_1 - U_0
    u2 = tl.cast(2.0, tl.float32) * y * u1 - u0
    result = tl.where(n_i32 == 2, u2, result)

    # k=3: U_3 = 2*y*U_2 - U_1
    u3 = tl.cast(2.0, tl.float32) * y * u2 - u1
    result = tl.where(n_i32 == 3, u3, result)

    # k=4: U_4 = 2*y*U_3 - U_2
    u4 = tl.cast(2.0, tl.float32) * y * u3 - u2
    result = tl.where(n_i32 == 4, u4, result)

    # For k >= 5, continue the recurrence
    # This is still sequential but we're trying to avoid the loop-carried type issue
    uk = u4
    ukm1 = u3
    ukm2 = u2
    for k in range(5, 65):
        uk_new = tl.cast(2.0, tl.float32) * y * uk - ukm2
        result = tl.where(n_i32 == k, uk_new, result)
        ukm2 = ukm1
        ukm1 = uk
        uk = uk_new

    return result.to(x.dtype)


def special_shifted_chebyshev_polynomial_u(A, B):
    logger.debug("GEMS SHIFTED_CHEBYSHEV_POLYNOMIAL_U")
    return _shifted_chebyshev_u_func(A, B)


def special_shifted_chebyshev_polynomial_u_(A, B):
    logger.debug("GEMS SHIFTED_CHEBYSHEV_POLYNOMIAL_U_")
    _shifted_chebyshev_u_func(A, B, out0=A)
    return A