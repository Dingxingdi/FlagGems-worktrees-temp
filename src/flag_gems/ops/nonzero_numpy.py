import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def count_nonzero_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """Count non-zero elements in a tensor."""
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero, axis=0)
    tl.atomic_add(out_ptr, nonzero_count)


@libentry()
@triton.jit
def nonzero_kernel(x_ptr, output_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """Kernel to compute linear indices for non-zero elements (fully vectorized)."""
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = x != 0

    # Create vector of linear indices for this block
    linear_indices = block_start + tl.arange(0, BLOCK_SIZE)

    # Use tl.where to select only non-zero indices
    # For zero elements, set to -1 (will be filtered in Python)
    selected = tl.where(is_nonzero, linear_indices, -1)

    # Store all selected indices at once
    # We need to create a proper mask for the store
    store_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    store_mask = store_offsets < numel
    tl.store(output_ptr + store_offsets, selected, mask=store_mask)


def nonzero_numpy(x):
    """
    Returns the indices of non-zero elements in the input tensor.

    Similar to numpy's nonzero(), returns a list of tensors where each tensor
    contains indices along one dimension.
    """
    logger.debug("GEMS NONZERO_NUMPY")

    if x.numel() == 0:
        # Empty tensor case - return empty list
        return [torch.tensor([], dtype=torch.int64, device=x.device) for _ in range(x.ndim)]

    # First, count the number of non-zero elements using Triton kernel
    x_flat = x.contiguous().flatten()
    numel = x_flat.numel()

    # Allocate counter
    count = torch.zeros(1, dtype=torch.int64, device=x.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    count_nonzero_kernel[grid](x_flat, count, numel, BLOCK_SIZE=BLOCK_SIZE)

    nonzero_count = count[0].item()

    if nonzero_count == 0:
        # No non-zero elements - return empty indices
        return [torch.tensor([], dtype=torch.int64, device=x.device) for _ in range(x.ndim)]

    # Allocate output for linear indices (same size as input)
    linear_indices = torch.full((numel,), -1, dtype=torch.int64, device=x.device)

    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    nonzero_kernel[grid](x_flat, linear_indices, numel, BLOCK_SIZE=BLOCK_SIZE)

    # Filter out -1 (which means zero element)
    linear_indices = linear_indices[linear_indices != -1]

    # Sort linear indices to ensure consistent ordering
    linear_indices = torch.sort(linear_indices).values

    # Convert linear indices to multi-dimensional indices
    shape = x.shape
    ndim = x.ndim

    result = []
    # Compute strides for the original shape
    strides = [1]
    for dim in range(ndim - 1, 0, -1):
        strides.insert(0, strides[0] * shape[dim])

    # Convert each linear index to multi-dimensional indices
    remaining = linear_indices
    for dim in range(ndim):
        dim_indices = (remaining // strides[dim]).to(torch.int64)
        result.append(dim_indices)
        remaining = remaining % strides[dim]

    return result