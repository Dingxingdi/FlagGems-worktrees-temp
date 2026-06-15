import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def histc_kernel(
    inp_ptr,
    out_ptr,
    n_elements,
    bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(inp_ptr + offsets, mask=mask, other=0.0)

    # Convert to float32 for computation
    vals_fp32 = vals.to(tl.float32)
    min_val_fp32 = min_val.to(tl.float32)
    max_val_fp32 = max_val.to(tl.float32)

    # Check valid: within [min_val, max_val] and not NaN
    # NaN check: x != x (NaN is the only value that is not equal to itself)
    is_valid = (vals_fp32 >= min_val_fp32) & (vals_fp32 <= max_val_fp32) & (vals_fp32 == vals_fp32)

    # Compute bin index: floor((val - min) / bin_width) where bin_width = (max - min) / bins
    # Equivalent to: floor((val - min) * bins / (max - min)), clamped to [0, bins-1]
    range_val = max_val_fp32 - min_val_fp32
    # Avoid division by zero - if range is 0, all valid values go to the last bin
    raw_bin = tl.where(
        range_val != 0.0,
        (vals_fp32 - min_val_fp32) * bins / range_val,
        (bins - 1) * 1.0,
    )
    # Use floor to ensure proper rounding at bin boundaries
    bin_idx = tl.floor(raw_bin).to(tl.int64)
    # Clamp bin_idx to [0, bins-1]
    bin_idx = tl.where(bin_idx < 0, 0, bin_idx)
    bin_idx = tl.where(bin_idx >= bins, bins - 1, bin_idx)

    # Atomic add to the appropriate bin (only for valid values and within bounds)
    tl.atomic_add(out_ptr + bin_idx, 1.0, mask=mask & is_valid, sem="relaxed")


def histc(inp, bins=100, min=0, max=0, *, out=None):
    logger.debug("GEMS HISTC")

    inp = inp.contiguous()
    n_elements = inp.numel()

    if n_elements == 0:
        result = torch.zeros(bins, dtype=torch.float32, device=inp.device)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Determine min and max values
    if min == 0 and max == 0:
        # Use actual min/max from data, excluding NaN
        # Filter out NaN values using isfinite
        valid_mask = torch.isfinite(inp)
        if not valid_mask.any():
            # All values are NaN or inf, result is all zeros
            result = torch.zeros(bins, dtype=torch.float32, device=inp.device)
            if out is not None:
                out.copy_(result)
                return out
            return result
        valid_vals = inp[valid_mask]
        min_val = valid_vals.min().item()
        max_val = valid_vals.max().item()
    else:
        min_val = float(min)
        max_val = float(max)

    # Handle edge case where min == max
    if min_val == max_val:
        # All values are the same, count them if they equal min_val
        count = torch.sum((inp == min_val) & ~torch.isnan(inp)).to(torch.float32)
        result = torch.zeros(bins, dtype=torch.float32, device=inp.device)
        # PyTorch puts all elements in the middle bin when min == max
        result[bins // 2] = count
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Allocate output tensor
    result = torch.zeros(bins, dtype=torch.float32, device=inp.device)

    # Determine block size
    BLOCK_SIZE = 1024
    num_warps = 4
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(inp.device):
        histc_kernel[grid](
            inp,
            result,
            n_elements,
            bins,
            min_val,
            max_val,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    if out is not None:
        out.copy_(result)
        return out
    return result