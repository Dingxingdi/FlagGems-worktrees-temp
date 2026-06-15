import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _grouped_mm_kernel(
    A,
    B,
    C,
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
    """Grouped matrix multiply kernel.

    Performs C = A @ B where A is (M, K) and B is (K, N).
    """
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # Row-major ordering of pids
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Create masks for valid indices
    mask_m = rm < M
    mask_n = rn < N
    mask_k = rk < K

    # Compute accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_K)):
        # Load A tile: (BLOCK_M, BLOCK_K)
        a = tl.load(
            A + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0,
        )
        # Load B tile: (BLOCK_K, BLOCK_N)
        b = tl.load(
            B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn),
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0,
        )
        # Accumulate dot product
        acc += tl.dot(a, b, allow_tf32=False)

        rk += BLOCK_K
        mask_k = rk < K

    # Store result
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
        acc.to(C.dtype.element_ty),
        mask=(mask_m[:, None] & mask_n[None, :]),
    )


def _grouped_mm(a, b, scales=None):
    """Grouped matrix multiply operation.

    Performs matrix multiplication C = A @ B.

    Args:
        a: First input tensor (M x K)
        b: Second input tensor (K x N)
        scales: Optional scale tensor (unused, for API compatibility)

    Returns:
        Output tensor (M x N)
    """
    logger.debug("GEMS _GROUPED_MM")

    assert a.is_cuda, "Input tensor must be on CUDA device"
    assert b.is_cuda, "Input tensor must be on CUDA device"
    assert a.ndim == 2, f"Expected 2D tensor for a, got {a.ndim}D"
    assert b.ndim == 2, f"Expected 2D tensor for b, got {b.ndim}D"
    assert a.shape[1] == b.shape[0], f"Matrix dimensions incompatible: {a.shape} @ {b.shape}"

    M, K = a.shape
    _, N = b.shape

    # Allocate output in float32 for accumulation, then convert to input dtype
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _grouped_mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c.to(a.dtype)