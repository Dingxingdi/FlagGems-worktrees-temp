import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("pairwise_distance"),
    key=["M", "N"],
)
@triton.jit
def pairwise_distance_kernel(
    x1,
    x2,
    out,
    M,
    N,
    p,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    x1 = x1 + pid * N
    x2 = x2 + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(x1 + cols, mask, other=0.0).to(tl.float32)
        b = tl.load(x2 + cols, mask, other=0.0).to(tl.float32)
        diff = a - b + eps
        _sum += pow(tl.abs(diff), p)
    sum_val = tl.sum(_sum, axis=1)

    out_val = pow(sum_val, 1.0 / p)[:, None]
    tl.store(out, out_val, row_mask)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS PAIRWISE_DISTANCE")
    if x1.shape != x2.shape:
        raise ValueError(f"x1 and x2 must have the same shape, got {x1.shape} and {x2.shape}")

    if x1.numel() == 0:
        out_shape = list(x1.shape[:-1]) if not keepdim else list(x1.shape)
        if keepdim:
            out_shape[-1] = 1
        out = torch.empty(out_shape, dtype=x1.dtype, device=x1.device)
        out.zero_()
        return out

    dtype = x1.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"pairwise_distance not implemented for {dtype}")

    M = 1
    for dim in range(x1.ndim - 1):
        M *= x1.shape[dim]
    N = x1.shape[-1]

    # Always compute with keepdim=True, then squeeze if needed
    out_shape = list(x1.shape)
    out_shape[-1] = 1

    x1 = x1.contiguous()
    x2 = x2.contiguous()
    out = torch.empty(out_shape, dtype=dtype, device=x1.device)

    with torch_device_fn.device(x1.device):
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        pairwise_distance_kernel[grid](
            x1, x2, out, M, N, p, eps
        )

    if not keepdim:
        out = out.squeeze(-1)
    return out