import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


@triton.jit
def transform_cauchy_f32(u, median, sigma):
    # Cauchy distribution: X = median + sigma * tan(pi * (U - 0.5))
    # Use tl.math.sin/cos which may be more optimized
    pi = 3.141592653589793
    angle = pi * (u - 0.5)
    return median + sigma * (tl.math.sin(angle) / tl.math.cos(angle))


@triton.jit
def transform_cauchy_f64(u, median, sigma):
    pi = 3.141592653589793
    angle = pi * (u - 0.5)
    return median + sigma * (tl.math.sin(angle) / tl.math.cos(angle))


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_cauchy_kernel_f32(
    out_ptr, N, median, sigma, philox_seed, philox_offset, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)

    y0 = transform_cauchy_f32(uint_to_uniform_float(r0), median, sigma)
    y1 = transform_cauchy_f32(uint_to_uniform_float(r1), median, sigma)
    y2 = transform_cauchy_f32(uint_to_uniform_float(r2), median, sigma)
    y3 = transform_cauchy_f32(uint_to_uniform_float(r3), median, sigma)

    start = pid.to(tl.uint64) * BLOCK * 4
    off0 = start + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    tl.store(out_ptr + off0, y0, mask=off0 < N)
    tl.store(out_ptr + off1, y1, mask=off1 < N)
    tl.store(out_ptr + off2, y2, mask=off2 < N)
    tl.store(out_ptr + off3, y3, mask=off3 < N)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def fused_cauchy_kernel_f64(
    out_ptr, N, median, sigma, philox_seed, philox_offset, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)

    # For f64, pair up the uints to get better uniform float precision
    u0 = uint_to_uniform_float(r0) * 0.5 + uint_to_uniform_float(r2) * 0.5
    u1 = uint_to_uniform_float(r1) * 0.5 + uint_to_uniform_float(r3) * 0.5

    y0 = transform_cauchy_f64(u0, median, sigma)
    y1 = transform_cauchy_f64(u1, median, sigma)

    start = pid.to(tl.uint64) * BLOCK * 2
    off0 = start + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK

    tl.store(out_ptr + off0, y0, mask=off0 < N)
    tl.store(out_ptr + off1, y1, mask=off1 < N)


def cauchy_(self, median: float = 0.0, sigma: float = 1.0, *, generator=None):
    logger.debug("GEMS CAUCHY_")

    dtype = self.dtype
    device = self.device
    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    N = self.numel()
    if N == 0:
        return self

    if dtype is torch.float64:
        UNROLL = 2
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
        increment = triton.cdiv(N, UNROLL)
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=generator
        )
        with torch_device_fn.device(device):
            fused_cauchy_kernel_f64[grid](
                self, N, median, sigma, philox_seed, philox_offset
            )
    else:
        UNROLL = 4
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
        increment = triton.cdiv(N, UNROLL)
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=generator
        )
        with torch_device_fn.device(device):
            fused_cauchy_kernel_f32[grid](
                self, N, median, sigma, philox_seed, philox_offset
            )

    return self