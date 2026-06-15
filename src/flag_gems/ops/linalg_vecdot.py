import logging
import math
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def vecdot_dim_kernel_inner(
    output_ptr,
    input1_ptr,
    input2_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input1_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input1_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input1_ptr.dtype.element_ty

    pid_m = tle.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp1_offset = pid_m * N + n_offsets
        inp2_offset = pid_m * N + n_offsets
        input1_ptrs = input1_ptr + inp1_offset
        input2_ptrs = input2_ptr + inp2_offset
        mask = n_offsets < N
        inp1 = tl.load(input1_ptrs, mask=mask, other=0).to(cdtype)
        inp2 = tl.load(input2_ptrs, mask=mask, other=0).to(cdtype)
        out = tl.sum(inp1 * inp2, axis=0)
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)
    else:
        sum = tl.zeros(
            [
                TILE_N,
            ],
            dtype=cdtype,
        )
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp1_offsets = pid_m * N + n_offsets
            inp2_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp1 = tl.load(input1_ptr + inp1_offsets, mask=mask, other=0).to(cdtype)
            inp2 = tl.load(input2_ptr + inp2_offsets, mask=mask, other=0).to(cdtype)
            sum += inp1 * inp2
        out = tl.sum(sum, axis=0)
        out_offset = pid_m
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def vecdot_dim_kernel_non_inner(
    output_ptr,
    input1_ptr,
    input2_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    if tl.constexpr(input1_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input1_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input1_ptr.dtype.element_ty

    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp1_offset = pid_m * N * K + n_offsets * K + k_offsets
        inp2_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        input1_ptrs = input1_ptr + inp1_offset
        input2_ptrs = input2_ptr + inp2_offset
        inp1 = tl.load(input1_ptrs, mask=mask, other=0).to(cdtype)
        inp2 = tl.load(input2_ptrs, mask=mask, other=0).to(cdtype)
        out = tl.sum(inp1 * inp2, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)
    else:
        sum = tl.zeros([TILE_N, TILE_K], dtype=cdtype)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp1_offsets = pid_m * N * K + n_offsets * K + k_offsets
            inp2_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp1 = tl.load(input1_ptr + inp1_offsets, mask=mask, other=0).to(cdtype)
            inp2 = tl.load(input2_ptr + inp2_offsets, mask=mask, other=0).to(cdtype)
            sum += inp1 * inp2
        out = tl.sum(sum, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        output_ptrs = output_ptr + out_offset
        tl.store(output_ptrs, out, mask=k_offsets < K)


def vecdot(x, y, *, dim=-1):
    logger.debug("GEMS VECDOT")
    assert x.shape == y.shape, "Input tensors must have the same shape"

    dim = dim % x.ndim
    N = x.shape[dim]
    M = reduce(lambda a, b: a * b, x.shape[:dim], 1)
    K = x.numel() // M // N

    # Compute output shape
    shape = list(x.shape)
    shape[dim] = 1
    out_shape = tuple(shape[:dim] + shape[dim + 1:])

    x = x.contiguous()
    y = y.contiguous()

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    with torch_device_fn.device(x.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            vecdot_dim_kernel_non_inner[grid](
                out,
                x,
                y,
                M,
                N,
                K,
            )
        else:
            grid = (M, 1, 1)
            vecdot_dim_kernel_inner[grid](
                out,
                x,
                y,
                M,
                N,
            )

    return out