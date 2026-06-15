import logging
import math

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_layer_norm_kernel(
    input_ptr,  # pointer to the input
    residual_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    b_ptr,  # pointer to the bias
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = tle.program_id(0)
    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + cols * in_stride_c, mask, other=0.0).to(cdtype)
    r = tl.load(residual_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)

    # Add in float32 to avoid precision loss
    x = x + r

    # Compute mean and mean of squares for numerical stability
    # Using E[X^2] - E[X]^2 formula for variance
    mean = tl.sum(x, axis=0) / N
    mean_sq = tl.sum(x * x, axis=0) / N
    var = mean_sq - mean * mean
    rstd = 1 / tl.sqrt(var + eps)

    # Load weight and bias
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(cdtype)
    b = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(cdtype)

    # Apply layer norm: (x - mean) * rstd * weight + bias
    y = ((x - mean) * rstd * w + b).to(cdtype)

    # write back to residual first (modified in-place) - store x before layer norm
    tl.store(residual_ptr + cols * r_stride_c, x, mask=mask)

    # write back to input
    tl.store(input_ptr + cols * in_stride_c, y, mask=mask)


def fused_add_layer_norm(x, residual, normalized_shape, weight, bias=None, eps=1e-5):
    """
    This function performs fused residual addition and Layer normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.

    Args:
        x: Input tensor
        residual: Residual tensor to add (will be modified in-place)
        normalized_shape: Shape to normalize over
        weight: Layer norm weight
        bias: Layer norm bias (optional)
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (normalized_x, updated_residual)
    """
    logger.debug(
        "GEMS FUSED_ADD_LAYER_NORM FORWARD, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = triton.next_power_of_2(N)
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else bias

    with torch_device_fn.device(x.device):
        fused_add_layer_norm_kernel[M,](
            x, residual, weight, bias, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
    return x, residual


# Alias for Add+LayerNorm naming
add_layer_norm = fused_add_layer_norm