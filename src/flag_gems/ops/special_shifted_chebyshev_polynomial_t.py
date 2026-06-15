import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def shifted_chebyshev_polynomial_t_func(x, n):
    # Shifted Chebyshev polynomial of the first kind T_n^*(x) = T_n(2*x - 1)
    # Using recurrence relation:
    #   T_0^*(x) = 1
    #   T_1^*(x) = 2*x - 1
    #   T_{n+1}^*(x) = 2*(2*x - 1)*T_n^*(x) - T_{n-1}^*(x)
    x_f32 = x.to(tl.float32)
    n_i = n.to(tl.int32)

    # Compute (2*x - 1) once
    two_x_minus_1 = 2.0 * x_f32 - 1.0

    # Initialize result - start with T_0^* = 1 for all
    result = 1.0 + 0.0 * x_f32  # Creates a tensor of ones with same shape/dtype as x_f32

    # Handle n == 0 case: result = T_0^* = 1 (already set)
    # Handle n == 1 case: result = T_1^* = 2*x - 1
    result = tl.where(n_i == 1, two_x_minus_1, result)

    # For n >= 2, compute iteratively
    # Use a loop with max_iterations = 16 (covers most practical cases)
    prev_prev = result  # T_0^*
    prev = two_x_minus_1  # T_1^*
    current = result  # Will be overwritten

    MAX_DEGREE = 16
    for i in range(2, MAX_DEGREE + 1):
        current = 2.0 * two_x_minus_1 * prev - prev_prev
        prev_prev = prev
        prev = current
        # If n == i, this is our result
        result = tl.where(n_i == i, current, result)

    return result.to(x.dtype)


def special_shifted_chebyshev_polynomial_t(x, n):
    logger.debug("GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_T")
    return shifted_chebyshev_polynomial_t_func(x, n)


def special_shifted_chebyshev_polynomial_t_(x, n):
    logger.debug("GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_T_")
    shifted_chebyshev_polynomial_t_func(x, n, out0=x)
    return x