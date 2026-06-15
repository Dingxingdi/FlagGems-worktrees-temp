import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def dist_kernel_p2(
    inp1,
    inp2,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute squared difference for p=2 case."""
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp1_ptrs = inp1 + offset
    inp2_ptrs = inp2 + offset
    mask = offset < n_elements

    x1 = tl.load(inp1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(inp2_ptrs, mask=mask, other=0.0).to(tl.float32)
    diff = x1 - x2
    sq_diff = diff * diff

    # Use SquaredDifference for more accurate computation
    sum_val = tl.sum(sq_diff)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def dist_kernel_p1(
    inp1,
    inp2,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute absolute difference for p=1 case."""
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp1_ptrs = inp1 + offset
    inp2_ptrs = inp2 + offset
    mask = offset < n_elements

    x1 = tl.load(inp1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(inp2_ptrs, mask=mask, other=0.0).to(tl.float32)
    diff = x1 - x2
    abs_diff = tl.abs(diff)

    sum_val = tl.sum(abs_diff)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def dist_kernel_pinf(
    inp1,
    inp2,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute max absolute difference for p=inf case."""
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp1_ptrs = inp1 + offset
    inp2_ptrs = inp2 + offset
    mask = offset < n_elements

    x1 = tl.load(inp1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(inp2_ptrs, mask=mask, other=0.0).to(tl.float32)
    diff = x1 - x2
    abs_diff = tl.abs(diff)

    max_val = tl.max(abs_diff)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@libentry()
@triton.jit
def dist_kernel_general(
    inp1,
    inp2,
    mid,
    n_elements,
    mid_size,
    p_val,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute |diff|^p for general p case."""
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp1_ptrs = inp1 + offset
    inp2_ptrs = inp2 + offset
    mask = offset < n_elements

    x1 = tl.load(inp1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(inp2_ptrs, mask=mask, other=0.0).to(tl.float32)
    diff = x1 - x2
    abs_diff = tl.abs(diff)

    # Compute abs_diff^p using exp(log(abs_diff) * p)
    # For zero values, we need special handling
    safe_abs = tl.where(abs_diff == 0.0, 1.0, abs_diff)
    log_abs = tl.log(safe_abs)
    powered = tl.exp(log_abs * p_val)
    # Set powered to 0 where abs_diff was 0
    powered = tl.where(abs_diff == 0.0, 0.0, powered)

    sum_val = tl.sum(powered)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def dist_reduce_p2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Final reduction for p=2: sqrt of sum of squares."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    result = tl.sqrt(sum_val)
    tl.store(out, result)


@libentry()
@triton.jit
def dist_reduce_p1(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Final reduction for p=1: sum of absolute differences."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


@libentry()
@triton.jit
def dist_reduce_pinf(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """Final reduction for p=inf: max of absolute differences."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0).to(tl.float32)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


@libentry()
@triton.jit
def dist_reduce_general(mid, out, mid_size, p_val, BLOCK_MID: tl.constexpr):
    """Final reduction for general p: (sum)^(1/p)."""
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    # Compute sum^(1/p) = exp(log(sum) / p)
    safe_sum = tl.where(sum_val == 0.0, 1.0, sum_val)
    log_sum = tl.log(safe_sum)
    result = tl.exp(log_sum / p_val)
    # When sum_val is 0, result should be 0
    result = tl.where(sum_val == 0.0, 0.0, result)
    tl.store(out, result)


def dist(input, other, p=2.0):
    logger.debug("GEMS DIST")

    # Broadcast input and other to the same shape
    output_shape = torch.broadcast_shapes(input.shape, other.shape)
    input = input.broadcast_to(output_shape)
    other = other.broadcast_to(output_shape)

    input = input.contiguous()
    other = other.contiguous()

    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=input.device)
    out = torch.empty([], dtype=torch.float32, device=input.device)

    with torch_device_fn.device(input.device):
        if p == 2.0:
            dist_kernel_p2[(mid_size, 1, 1)](
                input, other, mid, n_elements, mid_size, block_size
            )
            dist_reduce_p2[(1, 1, 1)](mid, out, mid_size, block_mid)
        elif p == 1.0:
            dist_kernel_p1[(mid_size, 1, 1)](
                input, other, mid, n_elements, mid_size, block_size
            )
            dist_reduce_p1[(1, 1, 1)](mid, out, mid_size, block_mid)
        elif math.isinf(p):
            dist_kernel_pinf[(mid_size, 1, 1)](
                input, other, mid, n_elements, mid_size, block_size
            )
            dist_reduce_pinf[(1, 1, 1)](mid, out, mid_size, block_mid)
        else:
            dist_kernel_general[(mid_size, 1, 1)](
                input, other, mid, n_elements, mid_size, p, block_size
            )
            dist_reduce_general[(1, 1, 1)](mid, out, mid_size, p, block_mid)

    # Convert output to match input dtype
    if input.dtype in (torch.float16, torch.bfloat16):
        out = out.to(input.dtype)

    return out