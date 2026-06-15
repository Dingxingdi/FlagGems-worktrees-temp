import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)


@triton.jit
def safe_exp_f32(x):
    # Clamp x to avoid overflow/underflow
    min_val = x * 0.0 + -87.0
    max_val = x * 0.0 + 88.0
    x = tl.minimum(tl.maximum(x, min_val), max_val)
    return tl.exp(x)


@triton.jit
def safe_exp_f64(x):
    min_val = x * 0.0 + -708.0
    max_val = x * 0.0 + 709.0
    x = tl.minimum(tl.maximum(x, min_val), max_val)
    return tl.exp(x)


@triton.jit
def safe_sqrt_f32(x):
    return tl.sqrt(tl.maximum(x, x * 0.0 + 1e-10))


@triton.jit
def safe_sqrt_f64(x):
    return tl.sqrt(tl.maximum(x, x * 0.0 + 1e-10))


# Constants for Poisson sampling
# We use the "punched card" method which unrolls the loop
@triton.jit
def poisson_small_lambda(rate, philox_seed, philox_offset, idx):
    """
    Sample from Poisson distribution using the inverse transform method.
    For small rates (lambda < ~20).
    Uses unrolled loop approach: at each step, generate uniform and check if
    accumulated probability exceeds 1.
    """
    seed = philox_seed.to(tl.int64)
    offset = philox_offset.to(tl.int64)

    c0 = (offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    c0_idx = c0 + idx
    z = c0_idx * 0

    rate_f32 = rate.to(tl.float32)

    # Generate random numbers on demand
    # Algorithm: Start with S = exp(-λ), x = 0, then:
    # while S < U (uniform), x++, S += exp(-λ) * λ^x / x!
    # But we can simplify: S is cumulative probability P(X <= x)

    # Use direct method: count events in unit time with exponential inter-arrival
    # The number of events follows Poisson(λ)
    # We can sample exponential inter-arrival times and count events in [0,1]

    # exp(-rate) threshold
    threshold = safe_exp_f32(-rate_f32)

    # Use accumulated product method (inverse transform)
    # For Poisson, we need to find smallest n where:
    # prod(u_0, u_1, ..., u_n) < exp(-λ) * λ^n / n! * n!
    # which simplifies to checking cumulative probabilities

    # Simpler: Generate exponential waiting times and count events
    # Events ~ Poisson if waiting times are exponential(λ)
    # Count how many exponential(λ) samples sum to less than 1

    # But this still requires loops. Let's use a different approach.

    # For small λ, use the direct method with fixed unroll:
    # The probability mass function is P(X=n) = exp(-λ) * λ^n / n!
    # We can use the ratio: P(X=n+1) / P(X=n) = λ / (n+1)

    # Initialize cumulative probability
    # Start with P(X=0) = exp(-λ)
    p_x0 = threshold  # exp(-λ)

    # Generate uniforms to decide
    r0, r1, r2, r3 = tl.philox(seed, c0_idx, c1, z, z)
    u0 = uint_to_uniform_float(r0)
    u1 = uint_to_uniform_float(r1)
    u2 = uint_to_uniform_float(r2)
    u3 = uint_to_uniform_float(r3)

    # Start with x = 0, cumulative = P(X <= 0) = p_x0
    # P(X=n) = exp(-λ) * λ^n / n!
    # P(X <= n) = sum_{i=0}^{n} P(X=i)
    # We sample from the distribution by checking if U < P(X=0),
    # if not, check if U < P(X=0) + P(X=1), etc.

    # Use sequential probability method:
    # P(X > n) = exp(-λ) * sum_{k=n+1}^{∞} λ^k / k!
    # = exp(-λ) * λ^(n+1) / (n+1)! * (1 + λ/(n+2) + ...)
    # For small λ, probability of large n is very small

    # Compute cumulative probabilities for small n
    # P(X=0) = exp(-λ)
    # P(X=1) = exp(-λ) * λ
    # P(X=2) = exp(-λ) * λ^2 / 2
    # P(X=3) = exp(-λ) * λ^3 / 6
    # etc.

    # Let U be uniform. Find smallest n such that U < sum_{i=0}^{n} P(X=i)

    # Compute P(X=0) = exp(-λ)
    p0 = threshold

    # Compute P(X=1) = exp(-λ) * λ
    p1 = p0 * rate_f32

    # Compute P(X=2) = P(X=1) * λ / 2
    p2 = p1 * rate_f32 * 0.5

    # Compute P(X=3) = P(X=2) * λ / 3
    p3 = p2 * rate_f32 * 0.3333333333

    # And so on...
    p4 = p3 * rate_f32 * 0.25
    p5 = p4 * rate_f32 * 0.2
    p6 = p5 * rate_f32 * 0.1666666667
    p7 = p6 * rate_f32 * 0.1428571429
    p8 = p7 * rate_f32 * 0.125
    p9 = p8 * rate_f32 * 0.1111111111
    p10 = p9 * rate_f32 * 0.1
    p11 = p10 * rate_f32 * 0.0909090909
    p12 = p11 * rate_f32 * 0.0833333333
    p13 = p12 * rate_f32 * 0.0769230769
    p14 = p13 * rate_f32 * 0.0714285714
    p15 = p14 * rate_f32 * 0.0666666667
    p16 = p15 * rate_f32 * 0.0625
    p17 = p16 * rate_f32 * 0.0588235294
    p18 = p17 * rate_f32 * 0.0555555556
    p19 = p18 * rate_f32 * 0.0526315789

    # Cumulative probabilities
    cum0 = p0
    cum1 = cum0 + p1
    cum2 = cum1 + p2
    cum3 = cum2 + p3
    cum4 = cum3 + p4
    cum5 = cum4 + p5
    cum6 = cum5 + p6
    cum7 = cum6 + p7
    cum8 = cum7 + p8
    cum9 = cum8 + p9
    cum10 = cum9 + p10
    cum11 = cum10 + p11
    cum12 = cum11 + p12
    cum13 = cum12 + p13
    cum14 = cum13 + p14
    cum15 = cum14 + p15
    cum16 = cum15 + p16
    cum17 = cum16 + p17
    cum18 = cum17 + p18
    cum19 = cum18 + p19

    # Use a single random number to sample from the distribution
    # Find the smallest n where U < cum_n
    U = u0  # Use the first random number

    # Chain of conditional returns (Triton doesn't support break)
    # For small rates, most probability is in low n values
    result = tl.where(U < cum0, 0,
               tl.where(U < cum1, 1,
               tl.where(U < cum2, 2,
               tl.where(U < cum3, 3,
               tl.where(U < cum4, 4,
               tl.where(U < cum5, 5,
               tl.where(U < cum6, 6,
               tl.where(U < cum7, 7,
               tl.where(U < cum8, 8,
               tl.where(U < cum9, 9,
               tl.where(U < cum10, 10,
               tl.where(U < cum11, 11,
               tl.where(U < cum12, 12,
               tl.where(U < cum13, 13,
               tl.where(U < cum14, 14,
               tl.where(U < cum15, 15,
               tl.where(U < cum16, 16,
               tl.where(U < cum17, 17,
               tl.where(U < cum18, 18,
               tl.where(U < cum19, 19, 19))))))))))))))))))))

    return result


@triton.jit
def poisson_large_lambda(rate, philox_seed, philox_offset, idx):
    """
    Sample from Poisson distribution using normal approximation.
    For large rates (lambda >= ~20).
    Uses Box-Muller transform to generate normal random numbers.
    """
    seed = philox_seed.to(tl.int64)
    offset = philox_offset.to(tl.int64)

    c0 = (offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    c0_idx = c0 + idx
    z = c0_idx * 0
    r0, r1, r2, r3 = tl.philox(seed, c0_idx, c1, z, z)

    rate_f32 = rate.to(tl.float32)
    std = safe_sqrt_f32(rate_f32)

    # Box-Muller transform for two normal samples
    u1 = uint_to_uniform_float(r0)
    u2 = uint_to_uniform_float(r1)

    # Avoid log(0)
    u1 = tl.maximum(u1, u1 * 0.0 + 1e-7)
    z0 = tl.sqrt(-2.0 * tl.log(u1)) * tl.cos(2.0 * 3.14159265359 * u2)

    # Normal approximation: X ~ N(lambda, lambda)
    sample = rate_f32 + std * z0

    # Poisson samples must be >= 0
    sample = tl.maximum(sample, sample * 0.0)

    return tl.cast(sample, tl.int32)


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def poisson_kernel_f32(
    out_ptr,
    rate_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    n = i

    rate = tl.load(rate_ptr + n, mask=n < N, other=0.0)
    rate_f32 = rate.to(tl.float32)

    # Select algorithm based on rate (threshold = 20.0)
    use_small = rate_f32 < 20.0

    # Compute small and large lambda samples
    small_sample = poisson_small_lambda(rate, philox_seed, philox_offset, i)
    large_sample = poisson_large_lambda(rate, philox_seed, philox_offset, i)

    # Select based on lambda size
    sample = tl.where(use_small, small_sample, large_sample)

    tl.store(out_ptr + n, sample, mask=n < N)


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
def poisson_kernel_f64(
    out_ptr,
    rate_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    n = i

    rate = tl.load(rate_ptr + n, mask=n < N, other=0.0)
    rate_f64 = rate.to(tl.float64)

    # For f64, use higher threshold
    use_small = rate_f64 < 20.0

    seed = philox_seed.to(tl.int64)
    offset = philox_offset.to(tl.int64)
    c0 = (offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    c0_idx = c0 + i
    z = c0_idx * 0
    r0, r1, r2, r3 = tl.philox(seed, c0_idx, c1, z, z)

    # Small lambda path for f64
    threshold = safe_exp_f64(-rate_f64)

    # Compute probabilities for Poisson
    p0 = threshold
    p1 = p0 * rate_f64
    p2 = p1 * rate_f64 * 0.5
    p3 = p2 * rate_f64 * 0.3333333333
    p4 = p3 * rate_f64 * 0.25
    p5 = p4 * rate_f64 * 0.2
    p6 = p5 * rate_f64 * 0.1666666667
    p7 = p6 * rate_f64 * 0.1428571429
    p8 = p7 * rate_f64 * 0.125
    p9 = p8 * rate_f64 * 0.1111111111
    p10 = p9 * rate_f64 * 0.1
    p11 = p10 * rate_f64 * 0.0909090909
    p12 = p11 * rate_f64 * 0.0833333333
    p13 = p12 * rate_f64 * 0.0769230769
    p14 = p13 * rate_f64 * 0.0714285714
    p15 = p14 * rate_f64 * 0.0666666667
    p16 = p15 * rate_f64 * 0.0625
    p17 = p16 * rate_f64 * 0.0588235294
    p18 = p17 * rate_f64 * 0.0555555556
    p19 = p18 * rate_f64 * 0.0526315789

    cum0 = p0
    cum1 = cum0 + p1
    cum2 = cum1 + p2
    cum3 = cum2 + p3
    cum4 = cum3 + p4
    cum5 = cum4 + p5
    cum6 = cum5 + p6
    cum7 = cum6 + p7
    cum8 = cum7 + p8
    cum9 = cum8 + p9
    cum10 = cum9 + p10
    cum11 = cum10 + p11
    cum12 = cum11 + p12
    cum13 = cum12 + p13
    cum14 = cum13 + p14
    cum15 = cum14 + p15
    cum16 = cum15 + p16
    cum17 = cum16 + p17
    cum18 = cum17 + p18
    cum19 = cum18 + p19

    U = uint_to_uniform_float(r0)

    small_sample = tl.where(U < cum0, 0,
                   tl.where(U < cum1, 1,
                   tl.where(U < cum2, 2,
                   tl.where(U < cum3, 3,
                   tl.where(U < cum4, 4,
                   tl.where(U < cum5, 5,
                   tl.where(U < cum6, 6,
                   tl.where(U < cum7, 7,
                   tl.where(U < cum8, 8,
                   tl.where(U < cum9, 9,
                   tl.where(U < cum10, 10,
                   tl.where(U < cum11, 11,
                   tl.where(U < cum12, 12,
                   tl.where(U < cum13, 13,
                   tl.where(U < cum14, 14,
                   tl.where(U < cum15, 15,
                   tl.where(U < cum16, 16,
                   tl.where(U < cum17, 17,
                   tl.where(U < cum18, 18,
                   tl.where(U < cum19, 19, 19))))))))))))))))))))

    # Large lambda path for f64
    std = safe_sqrt_f64(rate_f64)
    u1 = uint_to_uniform_float(r1)
    u2 = uint_to_uniform_float(r2)
    u1 = tl.maximum(u1, u1 * 0.0 + 1e-7)
    z0 = tl.sqrt(-2.0 * tl.log(u1)) * tl.cos(2.0 * 3.14159265359 * u2)
    large_sample = rate_f64 + std * z0
    large_sample = tl.maximum(large_sample, large_sample * 0.0)
    large_sample = tl.cast(large_sample, tl.int32)

    sample = tl.where(use_small, small_sample, large_sample)

    tl.store(out_ptr + n, sample, mask=n < N)


def poisson(input, *, generator=None):
    logger.debug("GEMS POISSON")

    dtype = input.dtype
    device = input.device
    N = input.numel()

    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    # Input must be non-negative for Poisson
    # Rates must be non-negative
    out = torch.empty_like(input, dtype=torch.int64)

    philox_seed, philox_offset = philox_backend_seed_offset(N, generator=generator)

    if dtype == torch.float64:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
        with torch_device_fn.device(device):
            poisson_kernel_f64[grid](
                out,
                input,
                N,
                philox_seed,
                philox_offset,
            )
    else:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
        with torch_device_fn.device(device):
            poisson_kernel_f32[grid](
                out,
                input,
                N,
                philox_seed,
                philox_offset,
            )

    return out


def poisson_(input, *, generator=None):
    logger.debug("GEMS POISSON_")

    dtype = input.dtype
    device = input.device
    N = input.numel()

    assert dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    # Output is int64 for poisson
    out = torch.empty_like(input, dtype=torch.int64)

    philox_seed, philox_offset = philox_backend_seed_offset(N, generator=generator)

    if dtype == torch.float64:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
        with torch_device_fn.device(device):
            poisson_kernel_f64[grid](
                out,
                input,
                N,
                philox_seed,
                philox_offset,
            )
    else:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
        with torch_device_fn.device(device):
            poisson_kernel_f32[grid](
                out,
                input,
                N,
                philox_seed,
                philox_offset,
            )

    # Copy back to input
    input.copy_(out)
    return input