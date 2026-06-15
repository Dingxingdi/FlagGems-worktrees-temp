import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _compute_row_offsets(row, col, offset):
    """Compute cumulative offsets for each row."""
    row_offsets = [0]
    for i in range(row):
        col_end = min(col, max(i + offset + 1, 0))
        count = max(0, col_end)
        row_offsets.append(row_offsets[-1] + count)
    return row_offsets


@libentry()
@triton.jit
def tril_indices_kernel(
    output_ptr,
    row_offsets_ptr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generate tril_indices for a matrix using precomputed row offsets.

    For each output index, we find the corresponding (row_idx, col_idx) by:
    1. Finding which row the index belongs to (using row_offsets)
    2. Computing col_idx = idx - row_offsets[row_idx]
    """
    pid = tl.program_id(0)
    offset_idx = pid * BLOCK_SIZE
    stride = tl.num_programs(0) * BLOCK_SIZE

    for idx in range(offset_idx, num_elements, stride):
        linear_idx = idx + tl.arange(0, BLOCK_SIZE)
        mask = linear_idx < num_elements

        # For each index, we need to find which row it belongs to
        # We use a simple approach: check against each row's offset
        # This works well for typical matrix sizes (row <= 64)

        # Load first 16 row offsets
        r0 = tl.load(row_offsets_ptr + 0)
        r1 = tl.load(row_offsets_ptr + 1)
        r2 = tl.load(row_offsets_ptr + 2)
        r3 = tl.load(row_offsets_ptr + 3)
        r4 = tl.load(row_offsets_ptr + 4)
        r5 = tl.load(row_offsets_ptr + 5)
        r6 = tl.load(row_offsets_ptr + 6)
        r7 = tl.load(row_offsets_ptr + 7)
        r8 = tl.load(row_offsets_ptr + 8)
        r9 = tl.load(row_offsets_ptr + 9)
        r10 = tl.load(row_offsets_ptr + 10)
        r11 = tl.load(row_offsets_ptr + 11)
        r12 = tl.load(row_offsets_ptr + 12)
        r13 = tl.load(row_offsets_ptr + 13)
        r14 = tl.load(row_offsets_ptr + 14)
        r15 = tl.load(row_offsets_ptr + 15)

        # Check each row's range using comparisons
        in_row0 = linear_idx < r1
        in_row1 = (linear_idx >= r1) & (linear_idx < r2)
        in_row2 = (linear_idx >= r2) & (linear_idx < r3)
        in_row3 = (linear_idx >= r3) & (linear_idx < r4)
        in_row4 = (linear_idx >= r4) & (linear_idx < r5)
        in_row5 = (linear_idx >= r5) & (linear_idx < r6)
        in_row6 = (linear_idx >= r6) & (linear_idx < r7)
        in_row7 = (linear_idx >= r7) & (linear_idx < r8)
        in_row8 = (linear_idx >= r8) & (linear_idx < r9)
        in_row9 = (linear_idx >= r9) & (linear_idx < r10)
        in_row10 = (linear_idx >= r10) & (linear_idx < r11)
        in_row11 = (linear_idx >= r11) & (linear_idx < r12)
        in_row12 = (linear_idx >= r12) & (linear_idx < r13)
        in_row13 = (linear_idx >= r13) & (linear_idx < r14)
        in_row14 = (linear_idx >= r14) & (linear_idx < r15)
        in_row15 = linear_idx >= r15

        # Compute row_idx using where chains
        row_idx = tl.zeros_like(linear_idx)
        row_idx = tl.where(in_row0, 0, row_idx)
        row_idx = tl.where(in_row1, 1, row_idx)
        row_idx = tl.where(in_row2, 2, row_idx)
        row_idx = tl.where(in_row3, 3, row_idx)
        row_idx = tl.where(in_row4, 4, row_idx)
        row_idx = tl.where(in_row5, 5, row_idx)
        row_idx = tl.where(in_row6, 6, row_idx)
        row_idx = tl.where(in_row7, 7, row_idx)
        row_idx = tl.where(in_row8, 8, row_idx)
        row_idx = tl.where(in_row9, 9, row_idx)
        row_idx = tl.where(in_row10, 10, row_idx)
        row_idx = tl.where(in_row11, 11, row_idx)
        row_idx = tl.where(in_row12, 12, row_idx)
        row_idx = tl.where(in_row13, 13, row_idx)
        row_idx = tl.where(in_row14, 14, row_idx)
        row_idx = tl.where(in_row15, 15, row_idx)

        # Compute col_idx = linear_idx - row_offsets[row_idx]
        row_start = r0  # default
        row_start = tl.where(in_row0, r0, row_start)
        row_start = tl.where(in_row1, r1, row_start)
        row_start = tl.where(in_row2, r2, row_start)
        row_start = tl.where(in_row3, r3, row_start)
        row_start = tl.where(in_row4, r4, row_start)
        row_start = tl.where(in_row5, r5, row_start)
        row_start = tl.where(in_row6, r6, row_start)
        row_start = tl.where(in_row7, r7, row_start)
        row_start = tl.where(in_row8, r8, row_start)
        row_start = tl.where(in_row9, r9, row_start)
        row_start = tl.where(in_row10, r10, row_start)
        row_start = tl.where(in_row11, r11, row_start)
        row_start = tl.where(in_row12, r12, row_start)
        row_start = tl.where(in_row13, r13, row_start)
        row_start = tl.where(in_row14, r14, row_start)
        row_start = tl.where(in_row15, r15, row_start)

        col_idx = linear_idx - row_start

        # Store row indices at output_ptr[0:num_elements]
        tl.store(output_ptr + linear_idx, row_idx, mask=mask)

        # Store col indices at output_ptr[num_elements:2*num_elements]
        tl.store(output_ptr + num_elements + linear_idx, col_idx, mask=mask)


def tril_indices(row, col, offset=0, *, dtype=None, layout=None, device=None, pin_memory=None):
    """
    Generate lower triangular indices for a matrix of size (row, col).

    Returns a 2xN tensor where N is the number of elements in the lower triangle.
    The first row contains row indices, the second row contains column indices.
    """
    logger.debug("GEMS TRIL_INDICES")

    if dtype is None:
        dtype = torch.int64
    if device is None:
        device = torch.device("cuda")
    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False

    # Calculate the number of elements in the lower triangle
    # Condition: i >= j - offset => j <= i + offset
    # j ranges from 0 to min(col-1, i+offset)
    # So: col_start = 0, col_end = min(col, max(i + offset + 1, 0))
    num_elements = 0
    for i in range(row):
        col_end = min(col, max(i + offset + 1, 0))
        if col_end > 0:
            num_elements += col_end

    # Create output tensor: 2 x num_elements
    result = torch.empty((2, num_elements), dtype=dtype, device=device, layout=layout, pin_memory=pin_memory)

    if num_elements == 0:
        return result

    # Compute row offsets - pad to 16 entries for the kernel
    row_offsets = _compute_row_offsets(row, col, offset)
    while len(row_offsets) < 16:
        row_offsets.append(row_offsets[-1])
    row_offsets_tensor = torch.tensor(row_offsets[:16], dtype=dtype, device=device)

    # Launch kernel
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

    tril_indices_kernel[grid](
        result,
        row_offsets_tensor,
        num_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return result