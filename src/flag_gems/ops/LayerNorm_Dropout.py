import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def layer_norm_dropout_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps: tl.constexpr,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK_N: tl.constexpr,
):
    """
    Fused LayerNorm + Dropout kernel.
    """
    pid = tle.program_id(0)

    # Compute mean and variance for the row
    n_offsets = tl.arange(0, BLOCK_N)
    mask = n_offsets < N

    # Load input and compute mean
    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x, axis=0) / N

    # Compute variance
    d = x - m
    s = tl.where(mask, d * d, 0.0)
    sum_square = tl.sum(s, axis=0)
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    # Load weight and bias
    if weight_ptr is None:
        w = 1.0
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask).to(tl.float32)
    if bias_ptr is None:
        b = 0.0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask).to(tl.float32)

    # Compute LayerNorm output
    y = (x - m) * rstd * w + b

    # Generate random values for dropout
    # philox_seed and philox_offset are Python ints, convert them
    seed_i64 = philox_seed
    offset_i64 = philox_offset

    # Get lower and upper 32 bits
    c0 = (offset_i64 & 0xFFFFFFFF)
    c1 = ((offset_i64 >> 32) & 0xFFFFFFFF)
    c0 = c0 + pid
    c0_0 = c0 * 0

    r0, r1, r2, r3 = tl.philox(seed_i64, c0, c1, c0_0, c0_0)

    # Convert to uniform float
    rand0 = uint_to_uniform_float(r0)
    rand1 = uint_to_uniform_float(r1)
    rand2 = uint_to_uniform_float(r2)
    rand3 = uint_to_uniform_float(r3)

    # Use broadcasting to create random values for each position
    # Each position gets a different random value based on its position modulo 4
    rand_idx = (n_offsets & 3)
    rand_vals = tl.where(rand_idx == 0, rand0,
                tl.where(rand_idx == 1, rand1,
                tl.where(rand_idx == 2, rand2, rand3)))

    # Apply dropout mask
    dropout_mask = rand_vals > p
    y = y * tl.where(dropout_mask, scale, 0.0)

    # Store output (convert to original dtype)
    orig_dtype = in_ptr.dtype.element_ty
    y = y.to(orig_dtype)

    tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)


def layer_norm_dropout(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
    p=0.5,
    train=True,
):
    """
    Fused LayerNorm + Dropout operation.

    Args:
        input: Input tensor
        normalized_shape: Shape to normalize over
        weight: Optional weight for LayerNorm
        bias: Optional bias for LayerNorm
        eps: Epsilon for numerical stability
        p: Dropout probability
        train: Whether to apply dropout (training mode)

    Returns:
        Output tensor after LayerNorm and Dropout
    """
    logger.debug(
        "GEMS LAYERNORM_DROPOUT FORWARD, [input shape]: %s, [normalized_shape]: %s, [p]: %s",
        input.size(),
        normalized_shape,
        p,
    )

    # If not training or p == 0, just do LayerNorm without dropout
    if not train or p == 0.0:
        from flag_gems.ops.layernorm import layer_norm
        y, _, _ = layer_norm(input, normalized_shape, weight, bias, eps)
        return y

    N = math.prod(normalized_shape)
    M = input.numel() // N

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    y = torch.empty_like(input)

    # Compute dropout scale factor in Python
    dropout_scale = 1.0 / (1.0 - p)

    # Determine block size
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_N = max(64, min(BLOCK_N, 4096))  # Clamp between 64 and 4096

    with torch_device_fn.device(input.device):
        philox_seed, philox_offset = philox_backend_seed_offset(M)

        grid = (M, 1, 1)
        layer_norm_dropout_kernel[grid](
            input,
            y,
            weight,
            bias,
            M,
            N,
            eps,
            p,
            dropout_scale,
            philox_seed,
            philox_offset,
            BLOCK_N,
        )

    return y