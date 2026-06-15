import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def index_fill(input, dim, index, value):
    logger.debug("GEMS INDEX FILL")
    output = input.clone()
    return index_fill_(output, dim, index, value)


def index_fill_(inp, dim, index, value):
    logger.debug("GEMS INDEX FILL_")
    dim = dim if dim >= 0 else dim + inp.ndim
    assert 0 <= dim < inp.ndim, "Dimension out of range"
    assert (0 <= index).all() and (index < inp.size(dim)).all(), "Index out of range"

    # Extract scalar value from tensor if needed
    if isinstance(value, torch.Tensor):
        fill_value = value.item()
    else:
        fill_value = value

    # Use a different approach: compute fill positions using scatter-style logic
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    inp_numel = inp.numel()

    # Compute the linear offset for each element
    offsets = torch.arange(inp_numel, device=inp.device)
    dim_indices = (offsets // dim_stride) % dim_size

    # Build a flat mask for positions to fill
    fill_mask_flat = torch.zeros(inp_numel, dtype=torch.bool, device=inp.device)
    for idx_val in index:
        fill_mask_flat = fill_mask_flat | (dim_indices == idx_val.item())

    # Reshape mask to match inp shape
    fill_mask = fill_mask_flat.view(inp.shape)

    # Use in-place fill with masked assignment
    inp[fill_mask] = fill_value

    return inp