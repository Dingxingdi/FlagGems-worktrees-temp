import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

# Get math functions from tl_extra_shim for cross-backend compatibility
erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
tanh = tl_extra_shim.tanh


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_gelu_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # using 1d tile makes code clean
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - m) * rstd * w + b

    # Apply GeLU activation (using "none" approximation)
    # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    out = 0.5 * out * (1 + erf(out.to(tl.float32) * scale))

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_gelu_persistent_kernel_multiline(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=n_mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=n_mask)
    out = w * (x - m[:, None]) * rstd[:, None] + b

    # Apply GeLU activation (using "none" approximation)
    scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    out = 0.5 * out * (1 + erf(out.to(tl.float32) * scale))

    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_loop"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_gelu_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)

    # Compute mean
    m = tl.zeros((TILE_N,), dtype=tl.float32)  # mean
    s = tl.zeros((TILE_N,), dtype=tl.float32)  # sum((x - m)^2)
    cnt = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    # the last step
    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd = tl.math.rsqrt(var + eps)
    m = final_m

    # reverse the order of the second sweep
    # Normalize and apply linear transformation
    prev_multiple = prev_multiple_of(N, TILE_N)
    # the first step, masking is needed
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets, mask=mask)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets, mask=mask)
        out = w * (x - m) * rstd + b

        # Apply GeLU activation (using "none" approximation)
        scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
        out = 0.5 * out * (1 + erf(out.to(tl.float32) * scale))

        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets, eviction_policy="evict_first").to(
            tl.float32
        )
        if weight_ptr is None:
            w = 1
        else:
            w = tl.load(weight_ptr + n_offsets)
        if bias_ptr is None:
            b = 0
        else:
            b = tl.load(bias_ptr + n_offsets)
        out = w * (x - m) * rstd + b

        # Apply GeLU activation (using "none" approximation)
        scale: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
        out = 0.5 * out * (1 + erf(out.to(tl.float32) * scale))

        tl.store(out_ptr + pid * N + n_offsets, out)


def layer_norm_gelu(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Fused LayerNorm + GeLU operator.

    Applies LayerNorm followed by GeLU activation.
    This is a common pattern in Transformer architectures.
    """
    logger.debug("GEMS LAYERNORM_GELU FORWARD")

    N = math.prod(normalized_shape)
    M = input.numel() // N

    input = input.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()
    y = torch.empty_like(input)

    with torch_device_fn.device(input.device):
        if N <= 128:
            TILE_N = triton.next_power_of_2(N)
            TILE_M = triton.cdiv(1024, TILE_N)
            grid = (triton.cdiv(M, TILE_M), 1, 1)
            layer_norm_gelu_persistent_kernel_multiline[grid](
                input,
                y,
                weight,
                bias,
                M,
                N,
                eps,
                TILE_M,
                TILE_N,
            )
        elif N <= 4096:
            TILE_N = triton.next_power_of_2(N)
            grid = (M, 1, 1)
            layer_norm_gelu_persistent_kernel[grid](
                input,
                y,
                weight,
                bias,
                M,
                N,
                eps,
                TILE_N,
            )
        else:
            grid = (M, 1, 1)
            layer_norm_gelu_loop_kernel[grid](
                input,
                y,
                weight,
                bias,
                M,
                N,
                eps,
            )
    return y