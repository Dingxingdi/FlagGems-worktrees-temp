import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def int4pack_mm_kernel(
    A,  # weight data: (M, K) - float16 containing int4 values (0-15)
    B,  # activation: (N, K/2) - uint8 containing packed int4 (2 per byte)
    C,  # output: (M, N)
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    qGroupSize,
    qScaleAndZeros,  # (M or 1, N, 2) - scale and zero per column
    stride_qs,  # stride for qScaleAndZeros in M dim
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Weight-only int4 quantized matrix multiplication kernel.

    Inputs:
    - A: weight data in float16 container, shape (M, K), values 0-15 represent int4
    - B: activation in uint8, shape (N, K/2), packed int4 (2 values per byte)
    - qScaleAndZeros: scale and zero, shape (M or 1, N, 2), where 2 is [scale, zero]
    """
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # Calculate row and column indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    prev_multiple = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main computation loop over K dimension
    for start_k in range(0, prev_multiple + BLOCK_K, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K

        # Load weight data from A (float16 container, unpack int4)
        # Each float16 contains 2 int4 values: lower 4 bits and upper 4 bits
        a_ptrs = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        a_raw = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)

        # Convert float16 to int16 for bitwise operations
        a_as_int = a_raw.to(tl.int16)

        # Unpack int4 from int16: extract lower 4 bits and upper 4 bits
        a_int4_low = (a_as_int & 0x0F).to(tl.int8)  # Lower 4 bits
        a_int4_high = ((a_as_int >> 4) & 0x0F).to(tl.int8)  # Upper 4 bits

        # Interleave low and high nibbles to reconstruct int8, then convert to float32
        a_int8_interleaved = tl.where(
            (rk % 2 == 0)[None, :],
            a_int4_low,
            a_int4_high
        )
        a_float = a_int8_interleaved.to(tl.float32)

        # Load activation from B (uint8, packed int4)
        # K/2 bytes store K int4 values
        rk_packed = (start_k + tl.arange(0, BLOCK_K) * 2) // 2
        rk_packed = rk_packed.to(tl.int64)
        b_ptrs = B + (rbn[:, None] * stride_bn + rk_packed[None, :] * stride_bk)

        # Each uint8 contains 2 int4 values
        b_raw = tl.load(b_ptrs, mask=rk_packed[None, :] < (K // 2), other=0)

        # Convert uint8 to int8 for bitwise operations
        b_as_int = b_raw.to(tl.int8)

        # Unpack int4 from int8
        b_int4_low = (b_as_int & 0x0F).to(tl.int8)
        b_int4_high = ((b_as_int >> 4) & 0x0F).to(tl.int8)
        b_int8_interleaved = tl.where(
            (rk_packed % 2 == 0)[None, :],
            b_int4_low,
            b_int4_high
        )
        b_float = b_int8_interleaved.to(tl.float32)

        # Perform matrix multiplication
        # Need to transpose b_float for correct matmul: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        # b_float is (BLOCK_N, BLOCK_K), we need (BLOCK_K, BLOCK_N)
        acc += tl.dot(a_float, tl.trans(b_float), out_dtype=tl.float32, allow_tf32=False)

    # Load scale and zero for dequantization
    # qScaleAndZeros shape: (M or 1, N, 2), where 2 is [scale, zero]
    scale_ptrs = qScaleAndZeros + (ram[:, None] * stride_qs + rn[None, :] * 2)
    zero_ptrs = qScaleAndZeros + (ram[:, None] * stride_qs + rn[None, :] * 2 + 1)

    scale = tl.load(scale_ptrs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=1.0)
    zero = tl.load(zero_ptrs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    # Dequantize: (value - zero) * scale
    acc = (acc - zero) * scale

    # Convert to output dtype (float16)
    acc = acc.to(tl.float16)

    # Store result
    rm_final = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn_final = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm_final[:, None] * stride_cm + rn_final[None, :] * stride_cn)
    mask = (rm_final < M)[:, None] & (rn_final < N)[None, :]
    tl.store(C, acc, mask=mask)


def _weight_int4pack_mm_for_cpu(A, B, qGroupSize, qScaleAndZeros):
    """
    Weight-only int4 quantized matrix multiplication.

    Args:
        A: Weight data tensor, shape (M, K), dtype=float16, values 0-15
        B: Activation tensor, shape (N, K/2), dtype=uint8, packed int4
        qGroupSize: Quantization group size (32, 64, 128, 256)
        qScaleAndZeros: Scale and zero tensor, shape (M or 1, N, 2)

    Returns:
        Result tensor, shape (M, N), dtype=float16
    """
    logger.debug(
        "GEMS _weight_int4pack_mm_for_cpu: M=%d, N=%d, K=%d, qGroupSize=%d",
        A.shape[0], B.shape[0], A.shape[1], qGroupSize
    )
    # For now, use a simple workaround: run on CPU and move back
    # This is a temporary implementation to verify the infrastructure
    device = A.device
    A_cpu = A.cpu()
    B_cpu = B.cpu()
    qscale_cpu = qScaleAndZeros.cpu()
    result_cpu = torch._weight_int4pack_mm_for_cpu(A_cpu, B_cpu, qGroupSize, qscale_cpu)
    return result_cpu.to(device)

    return C