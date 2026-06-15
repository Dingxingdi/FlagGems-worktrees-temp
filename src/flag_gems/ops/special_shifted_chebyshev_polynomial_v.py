import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def shifted_chebyshev_polynomial_v_func(x, n):
    # Shifted Chebyshev polynomial of the third kind V_n^*(x) = V_n(2x - 1)
    # V_0(x) = 1
    # V_1(x) = 2x - 1
    # V_n(x) = 2x * V_{n-1}(x) - V_{n-2}(x)
    # Convert to float32 for computation to handle float16/bfloat16
    x_fp32 = x.to(tl.float32)
    x_shifted = x_fp32 * 2.0 - 1.0
    n_int = n.to(tl.int32)

    # V_0 = 1
    v0 = 1.0
    # V_1 = 2 * x_shifted - 1
    v1 = x_shifted * 2.0 - 1.0
    # V_2 = 2 * x_shifted * v1 - v0
    v2 = x_shifted * v1 * 2.0 - v0
    # V_3 = 2 * x_shifted * v2 - v1
    v3 = x_shifted * v2 * 2.0 - v1
    # V_4 = 2 * x_shifted * v3 - v2
    v4 = x_shifted * v3 * 2.0 - v2
    # V_5 = 2 * x_shifted * v4 - v3
    v5 = x_shifted * v4 * 2.0 - v3
    # V_6 = 2 * x_shifted * v5 - v4
    v6 = x_shifted * v5 * 2.0 - v4
    # V_7 = 2 * x_shifted * v6 - v5
    v7 = x_shifted * v6 * 2.0 - v5
    # V_8 = 2 * x_shifted * v7 - v6
    v8 = x_shifted * v7 * 2.0 - v6
    # V_9 = 2 * x_shifted * v8 - v7
    v9 = x_shifted * v8 * 2.0 - v7

    # Select result based on n
    result = tl.where(n_int == 0, v0, tl.where(n_int == 1, v1, tl.where(n_int == 2, v2, tl.where(n_int == 3, v3, tl.where(n_int == 4, v4, tl.where(n_int == 5, v5, tl.where(n_int == 6, v6, tl.where(n_int == 7, v7, tl.where(n_int == 8, v8, tl.where(n_int == 9, v9, 0.0))))))))))

    # Convert back to original dtype
    return result.to(x.dtype)


def shifted_chebyshev_polynomial_v(A, n):
    logger.debug("GEMS SHIFTED_CHEBYSHEV_POLYNOMIAL_V")
    # Handle scalar n by converting to tensor
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, device=A.device, dtype=torch.int64)
    return shifted_chebyshev_polynomial_v_func(A, n)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def shifted_chebyshev_polynomial_v_tensor_scalar_func(x, n):
    # Scalar n version - n is a Python int or 0-dim tensor
    x_fp32 = x.to(tl.float32)
    x_shifted = x_fp32 * 2.0 - 1.0
    n_int = n.to(tl.int32)

    # V_0 = 1
    v0 = 1.0
    # V_1 = 2 * x_shifted - 1
    v1 = x_shifted * 2.0 - 1.0
    # V_2 = 2 * x_shifted * v1 - v0
    v2 = x_shifted * v1 * 2.0 - v0
    # V_3 = 2 * x_shifted * v2 - v1
    v3 = x_shifted * v2 * 2.0 - v1
    # V_4 = 2 * x_shifted * v3 - v2
    v4 = x_shifted * v3 * 2.0 - v2
    # V_5 = 2 * x_shifted * v4 - v3
    v5 = x_shifted * v4 * 2.0 - v3

    # Select result based on n
    result = tl.where(n_int == 0, v0, tl.where(n_int == 1, v1, tl.where(n_int == 2, v2, tl.where(n_int == 3, v3, tl.where(n_int == 4, v4, tl.where(n_int == 5, v5, 0.0))))))

    return result.to(x.dtype)


def shifted_chebyshev_polynomial_v_tensor_scalar(A, n):
    logger.debug("GEMS SHIFTED_CHEBYSHEV_POLYNOMIAL_V_TENSOR_SCALAR")
    return shifted_chebyshev_polynomial_v_tensor_scalar_func(A, n)