import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def meshgrid_expand_kernel(output_ptr, input_ptr, expand_dim, expand_size, other_size, total_elements, BLOCK_SIZE: tl.constexpr):
    """Meshgrid kernel that expands one dimension.

    Args:
        output_ptr: output tensor pointer
        input_ptr: input 1D tensor pointer
        expand_dim: 0 for row expansion (output[row, col] = input[row]), 1 for col expansion (output[row, col] = input[col])
        expand_size: size of dimension being expanded (rows for dim=0, cols for dim=1)
        other_size: size of the other dimension
        total_elements: total number of elements in output
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    if expand_dim == 0:
        # Expand along rows: output[row, col] = input[row]
        row = offsets // other_size
        val = tl.load(input_ptr + row)
    else:
        # Expand along cols: output[row, col] = input[col]
        col = offsets % other_size
        val = tl.load(input_ptr + col)

    tl.store(output_ptr + offsets, val, mask=mask)


@libentry()
@triton.jit
def meshgrid_general_kernel(output_ptr, input_ptr, shape_ptr, strides_ptr, expand_dim, n_dims, total_elements, BLOCK_SIZE: tl.constexpr):
    """General n-dimensional meshgrid kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Decode flat offset to multi-dimensional indices
    remaining = offsets
    dim_idx0 = offsets * 0
    dim_idx1 = offsets * 0
    dim_idx2 = offsets * 0
    dim_idx3 = offsets * 0

    # Only support up to 4 dimensions for simplicity
    for d in range(n_dims):
        dim_size = tl.load(shape_ptr + d * 4)
        if dim_size > 1:
            idx = remaining % dim_size
            remaining = remaining // dim_size
            if d == 0:
                dim_idx0 = idx
            elif d == 1:
                dim_idx1 = idx
            elif d == 2:
                dim_idx2 = idx
            elif d == 3:
                dim_idx3 = idx

    # Get the index for the expand dimension
    if expand_dim == 0:
        expand_idx = dim_idx0
    elif expand_dim == 1:
        expand_idx = dim_idx1
    elif expand_dim == 2:
        expand_idx = dim_idx2
    else:
        expand_idx = dim_idx3

    # Load value from input
    val = tl.load(input_ptr + expand_idx)

    # Compute output offset using strides
    stride0 = tl.load(strides_ptr + 0 * 4)
    stride1 = tl.load(strides_ptr + 1 * 4)
    stride2 = tl.load(strides_ptr + 2 * 4)
    stride3 = tl.load(strides_ptr + 3 * 4)

    out_offset = dim_idx0 * stride0 + dim_idx1 * stride1
    if n_dims > 2:
        out_offset = out_offset + dim_idx2 * stride2
    if n_dims > 3:
        out_offset = out_offset + dim_idx3 * stride3

    tl.store(output_ptr + out_offset, val, mask=mask)


def meshgrid(tensors: List[torch.Tensor], indexing: Optional[str] = None) -> List[torch.Tensor]:
    """Generate grids of coordinates from 1D input tensors.

    Args:
        tensors: List of 1D tensors
        indexing: 'ij' or 'xy', defaults to 'ij'

    Returns:
        List of tensors, each with shape (S0, S1, ..., S_{n-1})
    """
    logger.debug("GEMS MESHGRID")

    if indexing is None:
        indexing = 'ij'

    if not tensors:
        return []

    n = len(tensors)

    # Handle scalar inputs (0D tensors) - treat as 1D of size 1
    processed_tensors = []
    for t in tensors:
        if t.dim() == 0:
            processed_tensors.append(t.reshape(1))
        else:
            processed_tensors.append(t)

    # Get sizes of each input
    sizes = [t.shape[0] for t in processed_tensors]

    # Compute output shape
    if indexing == 'ij':
        output_shape = tuple(sizes)
    elif indexing == 'xy':
        if n >= 2:
            output_shape = tuple(sizes[1], sizes[0]) + tuple(sizes[2:])
        else:
            output_shape = tuple(sizes)
    else:
        raise ValueError(f"Invalid indexing mode: {indexing}")

    if n == 1:
        # Single input: just reshape
        return [processed_tensors[0].reshape(output_shape)]

    # Create output tensors
    outputs = []
    for i, t in enumerate(processed_tensors):
        output = torch.empty(output_shape, dtype=t.dtype, device=t.device)
        outputs.append(output)

    total_elements = 1
    for s in output_shape:
        total_elements *= s

    BLOCK_SIZE = 128
    grid = triton.cdiv(total_elements, BLOCK_SIZE)

    if n == 2:
        m, n_size = sizes[0], sizes[1]

        if indexing == 'ij':
            # grid_x: expand along rows (dim 0) -> use row index
            # grid_y: expand along cols (dim 1) -> use col index
            meshgrid_expand_kernel[grid,](
                outputs[0], processed_tensors[0], 0, m, n_size, total_elements, BLOCK_SIZE
            )
            meshgrid_expand_kernel[grid,](
                outputs[1], processed_tensors[1], 1, n_size, m, total_elements, BLOCK_SIZE
            )
        else:  # xy
            # grid_x: expand along cols (dim 1) -> use col index
            # grid_y: expand along rows (dim 0) -> use row index
            meshgrid_expand_kernel[grid,](
                outputs[0], processed_tensors[0], 1, n_size, m, total_elements, BLOCK_SIZE
            )
            meshgrid_expand_kernel[grid,](
                outputs[1], processed_tensors[1], 0, m, n_size, total_elements, BLOCK_SIZE
            )
    else:
        # For n > 2, use general implementation
        shape_tensor = torch.tensor([int(s) for s in output_shape], dtype=torch.int32, device=tensors[0].device)
        strides_tensor = torch.tensor([int(s) for s in outputs[0].stride()], dtype=torch.int32, device=tensors[0].device)

        for dim, (inp, out) in enumerate(zip(processed_tensors, outputs)):
            expand_dim = dim
            if indexing == 'xy' and n >= 2:
                if dim == 0:
                    expand_dim = 1
                elif dim == 1:
                    expand_dim = 0

            meshgrid_general_kernel[grid,](
                out, inp, shape_tensor, strides_tensor, expand_dim, n, total_elements, BLOCK_SIZE
            )

    return outputs


def meshgrid_(tensor: torch.Tensor, indexing: Optional[str] = None) -> torch.Tensor:
    """In-place meshgrid (for single tensor input)."""
    logger.debug("GEMS MESHGRID_")
    return meshgrid([tensor], indexing=indexing)[0]