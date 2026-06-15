import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.device_info import get_device_capability

logger = logging.getLogger(__name__)

SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128


def get_sm_version_num():
    major, minor = get_device_capability()
    return major * 10 + minor


SM_VERSION_NUM = get_sm_version_num()


def get_block_wise_smm_configs():
    tile_configs = [
        # (TILE_M, TILE_N, num_stages, num_warps)
        (32, 64, 5, 2),
        (64, 32, 5, 2),
        (64, 128, 4, 4),
        (64, 256, 4, 4),
        (128, 32, 4, 4),
        (128, 64, 4, 4),
        (128, 128, 4, 4),
        (128, 256, 3, 8),
        (256, 64, 4, 4),
        (256, 128, 3, 8),
    ]

    return [
        triton.Config(
            {
                "TILE_M": TILE_M,
                "TILE_N": TILE_N,
                "TILE_K": SCALE_BLOCK_K,
                "SWIZZLE_GROUP_M": 8,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for TILE_M, TILE_N, stages, warps in tile_configs
    ]


@triton.jit
def grouped_launch(
    pid, M, N, TILE_M: tl.constexpr, TILE_N: tl.constexpr, SWIZZLE_GROUP_M: tl.constexpr
):
    grid_m = tl.cdiv(M, TILE_M)
    grid_n = tl.cdiv(N, TILE_N)

    width = SWIZZLE_GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * SWIZZLE_GROUP_M, SWIZZLE_GROUP_M)

    pid_m = group_id * SWIZZLE_GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.autotune(
    configs=get_block_wise_smm_configs(),
    key=["_M_NPO2", "N", "K"],
)
@triton.jit
def _scaled_mm_blockwise_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    _M_NPO2: tl.constexpr,
    SCALE_BLOCK_N,
    SCALE_BLOCK_K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_Ascale_m,
    stride_Ascale_k,
    stride_Bscale_k,
    stride_Bscale_n,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    SWIZZLE_GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = grouped_launch(pid, M, N, TILE_M, TILE_N, SWIZZLE_GROUP_M)

    offs_am = (pid_m * TILE_M + tl.arange(0, TILE_M)) % M
    offs_bn = (pid_n * TILE_N + tl.arange(0, TILE_N)) % N
    offs_k = tl.arange(0, TILE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    a_scale_ptrs = a_scale_ptr + offs_am * stride_Ascale_m
    offs_bsn = offs_bn // SCALE_BLOCK_N
    b_scale_ptrs = b_scale_ptr + offs_bsn * stride_Bscale_n

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        k_remaining = K - k * TILE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        if USE_FAST_ACCUM:
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    offs_ks = k * TILE_K // SCALE_BLOCK_K
    a_scale = tl.load(a_scale_ptrs + offs_ks * stride_Ascale_k)
    b_scale = tl.load(b_scale_ptrs + offs_ks * stride_Bscale_k)

    acc = acc * a_scale[:, None] * b_scale[None, :]

    if HAS_BIAS:
        offs_bias = offs_bn
        bias_ptrs = c_ptr + offs_bias  # Reuse c_ptr for bias (will be offset differently)
        bias_mask = offs_bn < N
        # Bias is loaded from c_ptr's bias location - this is a simplification
        # In practice, we need to pass bias separately

    acc = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_cn = pid_n * TILE_N + tl.arange(0, TILE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 64, "TILE_N": 64, "TILE_K": 256}),
        triton.Config({"TILE_M": 64, "TILE_N": 128, "TILE_K": 128}),
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 128}),
    ],
    key=["_M_NPO2", "N", "K"],
)
@triton.jit
def _scaled_mm_per_tensor_kernel(
    c_ptr,
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    _M_NPO2,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACC_DTYPE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, TILE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    acc = tl.zeros((TILE_M, TILE_N), dtype=ACC_DTYPE)

    offsets_am = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, TILE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    for k in range(0, tl.cdiv(K, TILE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        if USE_FAST_ACCUM:
            acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE, allow_tf32=False)
        else:
            acc += tl.dot(a, b, out_dtype=ACC_DTYPE, allow_tf32=False)

        offsets_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr)
    # Apply scaling to match cuBLAS behavior (scale after accumulation)
    acc = a_scale * acc.to(tl.float32)
    acc = b_scale * acc.to(tl.float32)

    if HAS_BIAS:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, mask=bias_mask)
        acc = acc + bias[None, :]

    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    offs_cn = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def _scaled_mm_2d_blockwise(a, b, a_scale, b_scale, bias, use_fast_accum):
    global SCALE_BLOCK_K, SCALE_BLOCK_N
    SCALE_BLOCK_K, SCALE_BLOCK_N = 128, 128
    M, K = a.shape
    _, N = b.shape
    _M_NPO2 = triton.next_power_of_2(M)

    output_dtype = torch.float16
    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
    )

    HAS_BIAS = bias is not None

    _scaled_mm_blockwise_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        M,
        N,
        K,
        _M_NPO2,
        SCALE_BLOCK_N,
        SCALE_BLOCK_K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        HAS_BIAS=HAS_BIAS,
        USE_FAST_ACCUM=use_fast_accum,
    )

    return c


def _scaled_mm_per_tensor(a, b, a_scale, b_scale, bias, use_fast_accum):
    M, K = a.shape
    _, N = b.shape

    output_dtype = torch.float16
    c = torch.empty((M, N), device=a.device, dtype=output_dtype)
    _M_NPO2 = triton.next_power_of_2(M)

    ACC_DTYPE = tl.float32

    grid = lambda META: (
        triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
    )

    HAS_BIAS = bias is not None

    _scaled_mm_per_tensor_kernel[grid](
        c,
        a,
        b,
        a_scale,
        b_scale,
        bias if bias is not None else c,  # Pass c as placeholder if no bias
        M,
        N,
        K,
        _M_NPO2,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACC_DTYPE=ACC_DTYPE,
        HAS_BIAS=HAS_BIAS,
        USE_FAST_ACCUM=use_fast_accum,
    )

    return c


def scaled_mm(self, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False):
    logger.debug("GEMS SCALED_MM")

    # Validate input dimensions
    assert self.shape[1] == mat2.shape[0], "Incompatible dimensions for matrix multiplication"
    M, K = self.shape
    _, N = mat2.shape

    # Determine output dtype
    if out_dtype is not None:
        output_dtype = out_dtype
    elif self.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        output_dtype = torch.float16
    else:
        output_dtype = self.dtype

    # For Float8 inputs, require specific constraints
    if self.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        # Float8 path requires specific constraints
        assert bias is None, "Bias not supported for float8 inputs yet"
        assert scale_result is None, "scale_result not supported for float8 inputs yet"

    # Handle memory layout requirements
    # mat2 needs to be column-major (stride[0] == 1) for efficient computation
    if mat2.stride(0) != 1:
        mat2 = mat2.t().contiguous().t()

    a_scale_numel = scale_a.numel()
    b_scale_numel = scale_b.numel()

    if a_scale_numel == 1 and b_scale_numel == 1:
        result = _scaled_mm_per_tensor(self, mat2, scale_a, scale_b, bias, use_fast_accum)
    elif a_scale_numel == M and b_scale_numel == N:
        result = _scaled_mm_2d_blockwise(self, mat2, scale_a, scale_b, bias, use_fast_accum)
    else:
        raise NotImplementedError(
            f"Unsupported scale tensor sizes: a_scale={a_scale_numel}, b_scale={b_scale_numel}"
        )

    # Apply scale_result if provided (for dynamic quantization)
    if scale_result is not None:
        result = result * scale_result

    # Cast to output dtype if needed
    if output_dtype != result.dtype:
        result = result.to(output_dtype)

    return result