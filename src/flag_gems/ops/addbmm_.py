import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("addbmm", pre_hook=None)
    if os.environ.get("USE_FLAGTUNE") == "1"
    else runtime.get_tuned_config("addbmm"),
    key=["M", "N", "K"],
    strategy=runtime.get_expand_config("addbmm")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("addbmm"))
@triton.jit(do_not_specialize=["alpha", "beta"])
def addbmm_kernel(
    batch1_ptr,
    batch2_ptr,
    output_ptr,
    input_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    batch_size,
    stride_b1b,
    stride_b1m,
    stride_b1k,
    stride_b2b,
    stride_b2k,
    stride_b2n,
    stride_om,
    stride_on,
    stride_im,
    stride_in,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    # This kernel computes: output = beta * input + alpha * sum(batch1_i @ batch2_i)
    # where batch1 is (batch, M, K) and batch2 is (batch, K, N)
    # output and input are (M, N)

    pidx = tle.program_id(0)
    pidy = tle.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = tle.num_programs(0)
        gridy = tle.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    # Initialize accumulator for sum of all batch matrix multiplications
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over all batches and accumulate
    for batch_idx in range(batch_size):
        # Get pointers for current batch
        b1_ptr = batch1_ptr + batch_idx * stride_b1b
        b2_ptr = batch2_ptr + batch_idx * stride_b2b

        offs_k = tl.arange(0, TILE_K)
        a_ptrs = b1_ptr + offs_m[:, None] * stride_b1m + offs_k[None, :] * stride_b1k
        b_ptrs = b2_ptr + offs_k[:, None] * stride_b2k + offs_n[None, :] * stride_b2n

        num_iters = tl.cdiv(K, TILE_K)

        for k_iter in range(num_iters):
            if DIVISIBLE_K:
                if DIVISIBLE_M:
                    mask_a = None
                else:
                    mask_a = mask_m[:, None]
                if DIVISIBLE_N:
                    mask_b = None
                else:
                    mask_b = mask_n[None, :]
            else:
                mask_k = offs_k < K
                if DIVISIBLE_M:
                    mask_a = mask_k[None, :]
                else:
                    mask_a = mask_m[:, None] & mask_k[None, :]
                if DIVISIBLE_N:
                    mask_b = mask_k[:, None]
                else:
                    mask_b = mask_k[:, None] & mask_n[None, :]

            a = tl.load(a_ptrs, mask=mask_a)
            b = tl.load(b_ptrs, mask=mask_b)
            accumulator += tl.dot(a, b, allow_tf32=False)

            offs_k = offs_k + TILE_K
            a_ptrs = a_ptrs + TILE_K * stride_b1k
            b_ptrs = b_ptrs + TILE_K * stride_b2k

    # Load input and compute final output
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    else:
        mask_c = True
        if not DIVISIBLE_M:
            mask_c &= offs_m[:, None] < M
        if not DIVISIBLE_N:
            mask_c &= offs_n[None, :] < N

    inp = tl.load(input_ptrs, mask=mask_c)
    # Compute: output = beta * input + alpha * accumulator
    out = accumulator * alpha + inp * beta
    o = out.to(inp.dtype)
    tl.store(output_ptrs, o, mask=mask_c)


def addbmm_(input, batch1, batch2, beta=1.0, alpha=1.0):
    logger.debug("GEMS ADDBMM_")
    assert batch1.shape[0] == batch2.shape[0], "Batch dimension mismatch"
    assert batch1.shape[2] == batch2.shape[1], "K dimension mismatch"
    assert input.shape[0] == batch1.shape[1], "M dimension mismatch"
    assert input.shape[1] == batch2.shape[2], "N dimension mismatch"

    batch, M, K = batch1.shape
    _, _, N = batch2.shape

    # Make inputs contiguous
    batch1 = batch1.contiguous()
    batch2 = batch2.contiguous()
    input = input.contiguous()

    # The operation is in-place on input
    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
    )
    with torch_device_fn.device(input.device):
        addbmm_kernel[grid_fn](
            batch1,
            batch2,
            input,  # output in-place
            input,  # input for beta multiplication
            alpha,
            beta,
            M,
            N,
            K,
            batch,
            batch1.stride(0),
            batch1.stride(1),
            batch1.stride(2),
            batch2.stride(0),
            batch2.stride(1),
            batch2.stride(2),
            input.stride(0),
            input.stride(1),
            input.stride(0),
            input.stride(1),
        )
    return input


def addbmm(input, batch1, batch2, beta=1.0, alpha=1.0):
    logger.debug("GEMS ADDBMM")
    # Non-in-place version: create a copy first
    output = input.clone()
    return addbmm_(output, batch1, batch2, beta, alpha)