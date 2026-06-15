import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_im,
    stride_in,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # input: (M, K) row-major, accessed as (BM, BK)
    input_ptrs = input_ptr + (offs_am[:, None] * stride_im + offs_k[None, :] * stride_in)

    # weight: (N, K) row-major, but accessed as (BK, BN) for weight.T matmul
    # weight.T[k, n] = weight[n, k] = weight + n*K + k
    # For loading as (BK, BN), we need offs_k to vary fastest
    weight_ptrs = weight_ptr + (offs_bn[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        weight_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N)

        input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)
        weight_vals = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        accumulator += tl.dot(input_vals, weight_vals, allow_tf32=False)

        input_ptrs += BLOCK_SIZE_K * stride_in
        weight_ptrs += BLOCK_SIZE_K * stride_wk

    if HAS_BIAS:
        offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        output_ptrs = output_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
        output_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)

        bias_ptrs = bias_ptr + offs_on[None, :]
        bias_vals = tl.load(bias_ptrs, mask=offs_on[None, :] < N, other=0.0)

        accumulator = accumulator + bias_vals

    output = accumulator.to(input_ptr.dtype.element_ty)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    output_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(output_ptrs, output, mask=output_mask)


def linear(input, weight, bias=None):
    input_shape = input.shape
    in_features = input_shape[-1]
    out_features = weight.shape[0]

    M = 1
    for dim in input_shape[:-1]:
        M *= dim
    N = out_features
    K = in_features

    logger.debug(
        "GEMS LINEAR, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[input column-major]: %s, [weight column-major]: %s",
        M,
        N,
        K,
        input.stride(0) == 1 if len(input_shape) > 1 else True,
        weight.stride(0) == 1,
    )

    input = input.reshape(M, K).contiguous()
    weight = weight.contiguous()

    # Output buffer is (M, N) - we'll reshape to original shape at the end
    output_2d = torch.empty((M, N), device=input.device, dtype=input.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    # Define block sizes for different M, N ranges
    if M <= 16 and N <= 16:
        block_m, block_n, block_k = 16, 16, 16
    elif M <= 32 and N <= 64:
        block_m, block_n, block_k = 32, 64, 32
    elif M <= 64 and N <= 128:
        block_m, block_n, block_k = 64, 128, 32
    elif M <= 128 and N <= 256:
        block_m, block_n, block_k = 128, 256, 64
    else:
        block_m, block_n, block_k = 128, 128, 32

    with torch_device_fn.device(input.device):
        linear_kernel[grid](
            input,
            weight,
            bias if bias is not None else torch.zeros((), device=input.device, dtype=input.dtype),
            output_2d,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            weight.stride(0),  # stride_wn = K (row stride of N, K matrix)
            weight.stride(1),  # stride_wk = 1 (col stride)
            output_2d.stride(0),
            output_2d.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
        )
    return output_2d.reshape(input_shape[:-1] + (out_features,))


def linear_out(input, weight, bias=None, *, out=None):
    input_shape = input.shape
    in_features = input_shape[-1]
    out_features = weight.shape[0]

    M = 1
    for dim in input_shape[:-1]:
        M *= dim
    N = out_features
    K = in_features

    logger.debug(
        "GEMS LINEAR_OUT, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[input column-major]: %s, [weight column-major]: %s",
        M,
        N,
        K,
        input.stride(0) == 1 if len(input_shape) > 1 else True,
        weight.stride(0) == 1,
    )

    input = input.reshape(M, K).contiguous()
    weight = weight.contiguous()

    # Output buffer is (M, N) - caller reshape if needed
    output_2d = torch.empty((M, N), device=input.device, dtype=input.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    if M <= 16 and N <= 16:
        block_m, block_n, block_k = 16, 16, 16
    elif M <= 32 and N <= 64:
        block_m, block_n, block_k = 32, 64, 32
    elif M <= 64 and N <= 128:
        block_m, block_n, block_k = 64, 128, 32
    elif M <= 128 and N <= 256:
        block_m, block_n, block_k = 128, 256, 64
    else:
        block_m, block_n, block_k = 128, 128, 32

    with torch_device_fn.device(input.device):
        linear_kernel[grid](
            input,
            weight,
            bias if bias is not None else torch.zeros((), device=input.device, dtype=input.dtype),
            output_2d,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            output_2d.stride(0),
            output_2d.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
        )

    if out is None:
        return output_2d.reshape(input_shape[:-1] + (out_features,))
    else:
        out.copy_(output_2d.reshape(input_shape[:-1] + (out_features,)))
        return out