import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _weight_int4pack_mm_kernel(
    input_ptr,
    weight_ptr,
    scales_ptr,
    zeros_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wm,
    stride_wn,
    stride_scales,
    stride_zeros,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Kernel for _weight_int4pack_mm_with_scales_and_zeros operation.

    Computes: output = (input @ weight.T) * scales + zeros
    where input is (M, K), weight is (N, K), weight.T is (K, N)

    Args:
        input: Input matrix (M, K) - row-major
        weight: Weight matrix (N, K) - row-major, we use weight.T (K, N)
        scales: Quantization scales (N,)
        zeros: Quantization zero points (N,)

    Output shape: (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Use max_contiguous for better memory access
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)

    # Calculate the largest multiple of BLOCK_K that is <= K
    prev_multiple_k = (K // BLOCK_K) * BLOCK_K

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop: process full BLOCK_K chunks
    for start_k in range(0, prev_multiple_k, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)

        # Load input block: input is (M, K), we need rows rm, columns rk
        # Shape: (BLOCK_M, BLOCK_K)
        a = tl.load(input_ptr + (ram[:, None] * stride_im + rk[None, :] * stride_ik))

        # Load weight.T block: weight.T is (K, N), we need rows rk, columns rn
        # weight.T[rk, rn] = weight[rn, rk] in row-major storage
        # Shape: (BLOCK_K, BLOCK_N)
        b = tl.load(weight_ptr + (rbn[None, :] * stride_wm + rk[:, None] * stride_wn))

        if a.dtype != b.dtype:
            a = a.to(output_ptr.dtype.element_ty)
            b = b.to(output_ptr.dtype.element_ty)

        # Compute dot product: a @ b = (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) = (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # Handle remaining K elements (loop peeling)
    rk = (prev_multiple_k + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K

    a = tl.load(
        input_ptr + (ram[:, None] * stride_im + rk[None, :] * stride_ik),
        mask=mask_k[None, :],
        other=0.0,
    )
    b = tl.load(
        weight_ptr + (rbn[None, :] * stride_wm + rk[:, None] * stride_wn),
        mask=mask_k[:, None],
        other=0.0,
    )

    if a.dtype != b.dtype:
        a = a.to(output_ptr.dtype.element_ty)
        b = b.to(output_ptr.dtype.element_ty)

    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # Load scales and zeros for this output row
    scales = tl.load(scales_ptr + rn)
    zeros = tl.load(zeros_ptr + rn)

    # Apply scale and zero: output = acc * scale + zero
    acc = acc * scales + zeros

    # Convert to output dtype
    acc = acc.to(output_ptr.dtype.element_ty)

    # Store result
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    output_ptrs = output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(output_ptrs, acc, mask=mask)


def _weight_int4pack_mm_with_scales_and_zeros(input, weight, scales, zeros):
    """
    Matrix multiplication with int4 quantized weight (simulated with float weight).

    This operator performs: output = (input @ weight.T) * scales + zeros
    This is a standard GEMM with post-processing scale and zero addition.
    For int4 weights, the weights would be dequantized before this operation.

    Args:
        input: Input tensor of shape (M, K)
        weight: Weight tensor of shape (N, K) - can be float16/bfloat16
        scales: Quantization scales of shape (N,)
        zeros: Quantization zero points of shape (N,)

    Returns:
        Output tensor of shape (M, N)
    """
    logger.debug("GEMS _weight_int4pack_mm_with_scales_and_zeros")

    M, K = input.shape
    N, K_weight = weight.shape

    # Check dimensions
    assert K == K_weight, f"Incompatible dimensions: input K={K}, weight K={K_weight}"
    assert scales.shape[0] == N, f"Incompatible scales: expected {N}, got {scales.shape[0]}"
    assert zeros.shape[0] == N, f"Incompatible zeros: expected {N}, got {zeros.shape[0]}"

    # Create output tensor - use input dtype
    output_dtype = input.dtype
    output = torch.empty((M, N), dtype=output_dtype, device=input.device)

    # Ensure inputs are contiguous
    input = input.contiguous()
    weight = weight.contiguous()

    # Grid definition
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _weight_int4pack_mm_kernel[grid](
        input,
        weight,
        scales,
        zeros,
        output,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        scales.stride(0),
        zeros.stride(0),
        output.stride(0),
        output.stride(1),
    )

    return output