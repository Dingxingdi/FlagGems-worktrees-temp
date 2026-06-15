import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y):
    count = count_x + count_y
    _count = tl.maximum(count, 1)
    mc_x = mean_x * count_x
    mc_y = mean_y * count_y
    mean = (mc_x + mc_y) / _count
    M = M_x + mc_x * mean_x + M_y + mc_y * mean_y - count * mean * mean
    return mean, count, M


@libentry()
@triton.jit(do_not_specialize=["correction"])
def std_mean_welford_kernel(
    X,
    Std,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Std = Std + pid
    Mean = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)

        count = _count + mask
        cnt = tl.maximum(count, 1)
        cur_mean = (_mean * _count + x) / cnt
        _acc += (x - cur_mean) * (x - _mean) * mask
        _mean = cur_mean
        _count = count

    mean, _, acc = tl.reduce((_mean, _count, _acc), axis=1, combine_fn=welford_func)
    var = acc / (N - correction)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    std_dev = std_dev[:, None]
    mean = mean[:, None]
    # Write std / mean
    tl.store(Std, std_dev.to(Std.dtype.element_ty), row_mask)
    tl.store(Mean, mean.to(Mean.dtype.element_ty), row_mask)


@libentry()
@triton.jit
def std_mean_kernel_1(
    X,
    Acc,
    Average,
    Count,
    N,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    pid = tle.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + offset
    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid
    mask = offset < N

    x = tl.load(X, mask, other=0.0).to(tl.float32)

    count = tl.sum(mask.to(tl.float32))
    average = tl.sum(x) / count
    acc = tl.sum(x * x) - count * average * average

    tl.store(Average, average)
    tl.store(Acc, acc)
    tl.store(Count, count)


@libentry()
@triton.jit(do_not_specialize=["correction"])
def std_mean_kernel_2(
    Acc,
    Average,
    Count,
    Std,
    Mean,
    N,
    correction,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc, mask, other=0.0).to(tl.float32)
    average = tl.load(Average, mask, other=0.0).to(tl.float32)
    count = tl.load(Count, mask, other=0.0).to(tl.float32)

    mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)

    var = nvar / (N - correction)
    std_dev = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(Mean, mean.to(Mean.dtype.element_ty))
    tl.store(Std, std_dev.to(Std.dtype.element_ty))


def std_mean(x, dim=None, *, correction=1, keepdim=False):
    logger.debug("GEMS STD MEAN")

    if dim is None:
        # Global reduction - reduce all dimensions
        N = x.numel()
        if N == 0 or N - correction <= 0:
            # Empty tensor or invalid correction
            out_std = torch.full([], float("nan"), device=x.device, dtype=x.dtype)
            out_mean = torch.full([], float("nan"), device=x.device, dtype=x.dtype)
            return out_std, out_mean

        shape = [1] * x.ndim
        std = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_N = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N)
        acc = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        average = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        count = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            std_mean_kernel_1[(BLOCK_NUM,)](
                x, acc, average, count, N, BLOCK_N=BLOCK_N
            )
            std_mean_kernel_2[(1,)](
                acc, average, count, std, mean, N, correction, BLOCK_NUM, BLOCK_N=BLOCK_NUM
            )

        if not keepdim:
            std = std.squeeze()
            mean = mean.squeeze()
        return std, mean
    else:
        # Dimension-specific reduction - similar to var_mean
        shape = list(x.shape)
        if isinstance(dim, int):
            dim = [dim]
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N

        std = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        # Use fixed block sizes for now
        BLOCK_M = 1
        BLOCK_N = 1024

        grid = lambda META: (triton.cdiv(M, BLOCK_M),)
        with torch_device_fn.device(x.device):
            std_mean_welford_kernel[grid](
                x, std, mean, M, N, correction, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )

        if not keepdim:
            std = std.squeeze(dim=dim)
            mean = mean.squeeze(dim=dim)
        return std, mean