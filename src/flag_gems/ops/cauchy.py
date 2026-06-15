import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

UNROLL = 4


@triton.jit
def transform_cauchy_f32(u, median, sigma):
    # Cauchy inverse CDF: x = median + sigma * tan(pi * (u - 0.5))
    # tan(x) = sin(x) / cos(x) since tl.tan doesn't exist
    pi = 3.141592653589793
    angle = pi * (u - 0.5)
    return median + sigma * (tl.sin(angle) / tl.cos(angle))


@triton.jit
def transform_cauchy_f64(u, median, sigma):
    # Cauchy inverse CDF: x = median + sigma * tan(pi * (u - 0.5))
    # tan(x) = sin(x) / cos(x) since tl.tan doesn't exist
    pi = 3.141592653589793
    angle = pi * (u - 0.5)
    return median + sigma * (tl.sin(angle) / tl.cos(angle))


configs = [
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=configs, key=["N"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def cauchy_kernel_f32(
    out_ptr,
    N,
    median,
    sigma,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    c0_0 = transform_cauchy_f32(r0, median, sigma)
    c0_1 = transform_cauchy_f32(r1, median, sigma)
    c0_2 = transform_cauchy_f32(r2, median, sigma)
    c0_3 = transform_cauchy_f32(r3, median, sigma)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, c0_0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, c0_1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, c0_2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, c0_3, mask=off_3 < N, eviction_policy="evict_first")


@triton.autotune(configs=configs, key=["N"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def cauchy_kernel_f64(
    out_ptr,
    N,
    median,
    sigma,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    c0_0 = transform_cauchy_f64(r0, median, sigma)
    c0_1 = transform_cauchy_f64(r1, median, sigma)
    c0_2 = transform_cauchy_f64(r2, median, sigma)
    c0_3 = transform_cauchy_f64(r3, median, sigma)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, c0_0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, c0_1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, c0_2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, c0_3, mask=off_3 < N, eviction_policy="evict_first")


def cauchy(self, median=0.0, sigma=1.0, *, generator=None):
    logger.debug("GEMS CAUCHY")
    N = volume(self.shape)
    device = self.device
    out = torch.empty_like(self, dtype=self.dtype)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    dtype = self.dtype
    if dtype == torch.float64:
        with torch_device_fn.device(device):
            cauchy_kernel_f64[grid_fn](
                out, N, float(median), float(sigma), philox_seed, philox_offset
            )
    else:
        with torch_device_fn.device(device):
            cauchy_kernel_f32[grid_fn](
                out, N, float(median), float(sigma), philox_seed, philox_offset
            )
    return out


def cauchy_(self, median=0.0, sigma=1.0, *, generator=None):
    logger.debug("GEMS CAUCHY_")
    N = volume(self.shape)
    device = self.device
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    dtype = self.dtype
    if dtype == torch.float64:
        with torch_device_fn.device(device):
            cauchy_kernel_f64[grid_fn](
                self, N, float(median), float(sigma), philox_seed, philox_offset
            )
    else:
        with torch_device_fn.device(device):
            cauchy_kernel_f32[grid_fn](
                self, N, float(median), float(sigma), philox_seed, philox_offset
            )
    return self