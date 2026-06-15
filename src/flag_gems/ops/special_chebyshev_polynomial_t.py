import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_cos = tl_extra_shim.cos
_acos = tl_extra_shim.acos
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def chebyshev_polynomial_t_func(x, n):
    # Chebyshev polynomial of the first kind T_n(x)
    # T_0(x) = 1
    # T_1(x) = x
    # For n < 6 or |x| > 1: use recurrence T_{n+1} = 2*x*T_n - T_{n-1}
    # For n >= 6 and |x| <= 1: use trig formula T_n(x) = cos(n * acos(x))

    x_f32 = x.to(tl.float32)
    n_int = n.to(tl.int32)

    # Base cases
    T0 = 1.0
    T1 = x_f32

    # Pre-compute x powers for recurrence
    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    x4 = x3 * x_f32
    x5 = x4 * x_f32

    # Explicit formulas for n = 2 to 5
    T2 = 2.0 * x2 - 1.0
    T3 = 4.0 * x3 - 3.0 * x_f32
    T4 = 8.0 * x4 - 8.0 * x2 + 1.0
    T5 = 16.0 * x5 - 20.0 * x3 + 5.0 * x_f32

    # Recurrence for n >= 6: T_{k+1} = 2*x*T_k - T_{k-1}
    T6 = 2.0 * x_f32 * T5 - T4
    T7 = 2.0 * x_f32 * T6 - T5
    T8 = 2.0 * x_f32 * T7 - T6
    T9 = 2.0 * x_f32 * T8 - T7
    T10 = 2.0 * x_f32 * T9 - T8

    # Select result based on n
    result = T0  # n == 0
    result = tl.where(n_int == 1, T1, result)
    result = tl.where(n_int == 2, T2, result)
    result = tl.where(n_int == 3, T3, result)
    result = tl.where(n_int == 4, T4, result)
    result = tl.where(n_int == 5, T5, result)
    result = tl.where(n_int == 6, T6, result)
    result = tl.where(n_int == 7, T7, result)
    result = tl.where(n_int == 8, T8, result)
    result = tl.where(n_int == 9, T9, result)
    result = tl.where(n_int == 10, T10, result)

    # For n > 10:
    # - If |x| <= 1: use trig formula (more stable)
    # - If |x| > 1: continue using recurrence
    abs_x = tl.abs(x_f32)
    use_trig = (n_int > 10) & (abs_x <= 1.0)

    # Trig formula: T_n(x) = cos(n * acos(x))
    acos_x = _acos(x_f32)
    trig_result = _cos(n_int.to(tl.float32) * acos_x)

    # For n > 10, select based on |x|
    result = tl.where(use_trig, trig_result, result)

    return result


def chebyshev_polynomial_t(x, n):
    logger.debug("GEMS CHEBYSHEV_POLYNOMIAL_T")
    return chebyshev_polynomial_t_func(x, n)


def chebyshev_polynomial_t_(x, n):
    logger.debug("GEMS CHEBYSHEV_POLYNOMIAL_T_")
    return chebyshev_polynomial_t_func(x, n, out0=x)