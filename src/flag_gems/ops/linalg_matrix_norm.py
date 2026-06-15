import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

sqrt = tl_extra_shim.sqrt
logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("vector_norm"), key=["M", "N"])
@triton.jit
def fro_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a * a
    sum = tl.sum(_sum, axis=1)

    out = sqrt(sum)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def fro_norm_kernel_1(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(x * x)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def fro_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = sqrt(tl.sum(mid))
    tl.store(Out, out)


def linalg_matrix_norm(x, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    logger.debug("GEMS LINALG_MATRIX_NORM")

    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        elif not isinstance(dtype, torch.dtype):
            dtype = torch.float32
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"linalg_matrix_norm not implemented for {dtype}")

    # Handle ord string values
    if isinstance(ord, str):
        ord = ord.lower()

    # Only Frobenius norm is implemented in Triton
    # Other norms fall back to PyTorch (outside of use_gems context)
    if ord != "fro":
        raise NotImplementedError(
            f"linalg_matrix_norm with ord={ord} is not implemented in GEMS. "
            f"Only 'fro' norm is supported."
        )

    # Frobenius norm via Triton
    with torch_device_fn.device(x.device):
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N

        out = torch.empty(shape, dtype=dtype, device=x.device)

        if M == 1:
            # Single matrix case - use BLOCK_SIZE = N to process all N elements
            BLOCK_SIZE = triton.next_power_of_2(N)
            MID_SIZE = triton.cdiv(N, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            mid = torch.empty([MID_SIZE], dtype=dtype, device=x.device)

            fro_norm_kernel_1[(MID_SIZE,)](x, mid, N, BLOCK_SIZE)
            fro_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
        else:
            # Batch of matrices case
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
            fro_norm_kernel[grid](x, out, M, N)

    if not keepdim:
        out = out.squeeze(dim=dim)
    return out