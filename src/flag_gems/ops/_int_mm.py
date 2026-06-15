import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _int_mm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_ptrs = O + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    o_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)

    tl.store(o_ptrs, accumulator, mask=o_mask)


def _int_mm(A, B):
    logger.debug("GEMS _int_mm")
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matmul"
    M, K = A.shape
    K_, N = B.shape
    out = torch.empty((M, N), dtype=torch.int32, device=A.device)

    def grid_fn(meta):
        return (
            triton.cdiv(meta["M"], meta["BLOCK_SIZE_M"]),
            triton.cdiv(meta["N"], meta["BLOCK_SIZE_N"]),
        )

    with torch_device_fn.device(A.device):
        _int_mm_kernel[grid_fn](
            A,
            B,
            out,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=32,
        )
    return out