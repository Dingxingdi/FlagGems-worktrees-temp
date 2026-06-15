import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def dyn_quant_matmul_4bit_kernel(
    A,
    W,
    W_scale,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_cm,
    stride_cn,
    stride_ws_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for dynamic quantized 4-bit matmul.

    A: activation matrix (M, K) in float16/bfloat16
    W: quantized weight matrix (K/2, N) in int8 (packed int4)
    W_scale: scale matrix (N,) in float32
    C: output matrix (M, N) in float16/bfloat16
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Calculate offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for A: (M, K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # Pointers for W: (K/2, N) - packed int4, 2 values per byte
    # W is stored as (K/2, N) in int8
    # For each k, we load 2 consecutive int4 values
    # offs_k // 2 gives the column in W
    w_ptrs_base = W + ((offs_k[:, None] // 2) * stride_wk + offs_bn[None, :] * stride_wn)

    # Pointers for W_scale: (N,)
    w_scale_ptrs = W_scale + offs_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_offs = k_start + offs_k

        # Load A: (BLOCK_M, BLOCK_K)
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - k_start,
            other=0.0,
        ).to(tl.float32)

        # Load W: (BLOCK_K/2, BLOCK_N) in int8
        # Need to unpack int4 from int8
        w_packed = tl.load(
            w_ptrs_base,
            mask=(offs_k[:, None] // 2) < (K // 2) - (k_start // 2),
            other=0.0,
        )

        # Unpack int4 from int8: even indices use lower 4 bits, odd indices use upper 4 bits
        # For even k: (w_packed & 0xF) - 8 to convert to signed int4 range [-8, 7]
        # For odd k: (w_packed >> 4) - 8
        is_even = (k_offs % 2 == 0)[:, None]
        w_low = (w_packed & 0xF).to(tl.int8) - 8
        w_high = ((w_packed >> 4) & 0xF).to(tl.int8) - 8
        w_int4 = tl.where(is_even, w_low, w_high)

        # Convert to float and apply scale
        w_dequant = w_int4.to(tl.float32)
        w_scale = tl.load(w_scale_ptrs).to(tl.float32)
        w_scaled = w_dequant * w_scale[None, :]

        # Matrix multiplication
        accumulator += tl.dot(a, w_scaled, out_dtype=tl.float32)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs_base += (BLOCK_SIZE_K // 2) * stride_wk

    # Convert output to the target dtype
    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    # Store output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def dyn_quant_matmul_4bit(
    A: torch.Tensor,
    W: torch.Tensor,
    W_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dynamic quantized 4-bit matrix multiplication.

    Performs matmul(A, W_dequantized) where W is dequantized from int4 using W_scale.

    Args:
        A: Activation tensor of shape (M, K) in float16/bfloat16/float32
        W: Quantized weight tensor of shape (K//2, N) in int8 (packed int4)
        W_scale: Scale tensor of shape (N,) in float32
        output_dtype: Output dtype (default: float16)

    Returns:
        Output tensor of shape (M, N) in output_dtype
    """
    logger.debug("GEMS DYN_QUANT_MATMUL_4BIT")

    M, K = A.shape
    N = W.shape[1]
    assert K % 2 == 0, f"K dimension must be even, got {K}"
    assert W.shape[0] == K // 2, f"W dimension mismatch: expected {K // 2}, got {W.shape[0]}"
    assert W_scale.shape[0] == N, f"Scale dimension mismatch: expected {N}, got {W_scale.shape[0]}"

    # Ensure contiguous
    A = A.contiguous()
    W = W.contiguous()
    W_scale = W_scale.contiguous()

    # Allocate output
    C = A.new_empty((M, N), dtype=output_dtype)

    # Define kernel configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 128
    GROUP_SIZE_M = 8

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    with torch_device_fn.device(A.device):
        dyn_quant_matmul_4bit_kernel[grid](
            A,
            W,
            W_scale,
            C,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            W.stride(0),
            W.stride(1),
            C.stride(0),
            C.stride(1),
            W_scale.stride(0),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )

    return C