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


@triton.jit
def transform_geometric_f32(u, log_one_minus_p):
    # Inverse transform sampling for geometric distribution
    # X = floor(log(1 - u) / log(1 - p)) + 1
    # Since u is uniform in (0, 1), (1 - u) is also uniform in (0, 1)
    one_minus_u = 1.0 - u
    # Avoid log(0) by using a small threshold
    eps = 1e-7
    one_minus_u = tl.where(one_minus_u < eps, eps, one_minus_u)
    log_one_minus_u = tl.log(one_minus_u)
    # log(1 - u) / log(1 - p) is negative since both logs are negative
    # floor of a negative number goes toward more negative
    k = log_one_minus_u / log_one_minus_p
    return tl.floor(k) + 1.0


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def geometric_kernel(
    out_ptr,
    N,
    log_one_minus_p,
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
    r0 = transform_geometric_f32(r0, log_one_minus_p)
    r1 = transform_geometric_f32(r1, log_one_minus_p)
    r2 = transform_geometric_f32(r2, log_one_minus_p)
    r3 = transform_geometric_f32(r3, log_one_minus_p)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")


UNROLL = 4
BLOCK_SIZE = 1024


def geometric_(self, p: float = 1.0, *, generator=None):
    logger.debug("GEMS GEOMETRIC_")
    N = volume(self.shape)
    log_one_minus_p = float(torch.math.log(1.0 - p))
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        geometric_kernel[grid_fn](
            self, N, log_one_minus_p, philox_seed, philox_offset, BLOCK=BLOCK_SIZE
        )
    return self