import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _weight_int8pack_mm_kernel(
    A,  # int8 weight matrix [M, K]
    B,  # activation matrix [K, N]
    C,  # output matrix [M, N]
    scales,  # scaling factors [N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Weight-only int8 matrix multiplication kernel.

    Computes: C = (A @ B) * scales
    where:
    - A: int8 weight matrix [M, K]
    - B: activation matrix [K, N] (float16/bfloat16)
    - C: output matrix [M, N]
    - scales: per-column scaling factors [N]
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m - first_pid_m, num_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute offsets
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Load scales for this block
    scale_ptrs = scales + offs_bn
    scales_block = tl.load(scale_ptrs)

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main computation loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks
        a = tl.load(
            a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0
        )

        # Convert int8 to float32 for computation
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Matrix multiplication
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply scaling
    accumulator = accumulator * scales_block[None, :]

    # Convert to output dtype
    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@libentry()
@triton.autotune(
    configs=[
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _weight_int8pack_mm_kernel_autotuned(
    A,
    B,
    C,
    scales,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m - first_pid_m, num_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    scale_ptrs = scales + offs_bn
    scales_block = tl.load(scale_ptrs)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0
        )

        a = a.to(tl.float32)
        b = b.to(tl.float32)
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    accumulator = accumulator * scales_block[None, :]

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _weight_int8pack_mm(A: torch.Tensor, B: torch.Tensor, scales: torch.Tensor):
    """
    Weight-only int8 matrix multiplication.

    Performs: C = (A @ B.T) * scales (row-major) or C = (A @ B) * scales (col-major)

    Args:
        A: int8 weight matrix [M, K]
        B: activation matrix [K, N] (float16 or bfloat16)
        scales: scaling factors [N]

    Returns:
        Output matrix [M, N] with same dtype as B
    """
    logger.debug("GEMS _weight_int8pack_mm")
    assert A.dtype == torch.int8, f"Expected int8 weight, got {A.dtype}"
    assert B.dtype in [torch.float16, torch.bfloat16, torch.float32], f"Expected float16/bfloat16/float32 activation, got {B.dtype}"
    assert scales.dtype == B.dtype, f"Scales dtype {scales.dtype} must match activation dtype {B.dtype}"

    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, f"Incompatible dimensions: A.shape[1]={K} != B.shape[0]={K_B}"
    assert scales.shape[0] == N, f"Scales shape {scales.shape} must match B.shape[1]={N}"

    # Handle non-contiguous inputs
    if A.stride(0) > 1 and A.stride(1) > 1:
        A = A.contiguous()
    if B.stride(0) > 1 and B.stride(1) > 1:
        B = B.contiguous()

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=B.dtype)

    # Grid computation
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _weight_int8pack_mm_kernel_autotuned[grid](
        A,
        B,
        C,
        scales,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
    )

    return C