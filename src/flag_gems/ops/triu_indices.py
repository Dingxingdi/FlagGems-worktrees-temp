import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def triu_indices_kernel(
    indices_ptr,
    row_offsets_ptr,
    row,
    col,
    offset,
    count,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for generating triu_indices.

    For each output index k, we need to find the (row_idx, col_idx) pair:
    - row_idx: which row in the matrix
    - col_idx: which column in that row

    We use precomputed row_offsets to determine which row each index belongs to.
    """
    pid = tle.program_id(0)
    start_idx = pid * BLOCK_SIZE

    # Compute indices for this block
    idx_offset = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = idx_offset < count

    # For each index, compute the row and column
    # We need to find the row such that row_offsets[row] <= idx < row_offsets[row+1]
    # Since row is typically small, we can do a linear search

    row_idx = tl.zeros([BLOCK_SIZE], tl.int64)
    col_idx = tl.zeros([BLOCK_SIZE], tl.int64)

    for i in range(BLOCK_SIZE):
        idx = start_idx + i
        if idx >= count:
            break

        # Linear search to find the row
        # row_offsets[j] gives the starting index of row j
        for r in range(row):
            row_start = tl.load(row_offsets_ptr + r)
            row_end = tl.load(row_offsets_ptr + r + 1) if r + 1 <= row else count

            if idx >= row_start and idx < row_end:
                row_idx = r
                # Column is: max(0, r + offset) + (idx - row_start)
                start_col = r + offset
                if start_col < 0:
                    start_col = 0
                col_idx = start_col + (idx - row_start)
                break

    # Store results
    tl.store(indices_ptr + idx_offset, row_idx, mask=mask)
    tl.store(indices_ptr + count + idx_offset, col_idx, mask=mask)


def _compute_row_offsets(row, col, offset):
    """Compute the starting index for each row in the flattened triu indices."""
    offsets = []
    current = 0
    for i in range(row + 1):
        offsets.append(current)
        if i < row:
            start_col = i + offset
            if start_col < 0:
                start_col = 0
            if start_col < col:
                current += col - start_col
    return offsets


def _triu_indices_cpu(row, col, offset, dtype, device):
    """CPU fallback for computing triu_indices."""
    row = int(row)
    col = int(col)
    offset = int(offset)

    row_indices = []
    col_indices = []

    for i in range(row):
        start_col = i + offset
        if start_col < 0:
            start_col = 0
        for j in range(start_col, col):
            row_indices.append(i)
            col_indices.append(j)

    row_tensor = torch.tensor(row_indices, dtype=dtype)
    col_tensor = torch.tensor(col_indices, dtype=dtype)

    return torch.stack([row_tensor, col_tensor])


def triu_indices(row, col, offset=0, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS TRIU_INDICES")

    row = int(row)
    col = int(col)
    offset = int(offset)

    if dtype is None:
        dtype = torch.int64

    if device is None:
        device = runtime.device.name

    if pin_memory is None:
        pin_memory = False

    # Compute the number of elements in the upper triangular part
    count = 0
    for i in range(row):
        start_col = i + offset
        if start_col < 0:
            start_col = 0
        if start_col < col:
            count += col - start_col

    # Handle empty case
    if count == 0:
        return torch.empty((2, 0), dtype=dtype, device=device, pin_memory=pin_memory)

    # Create output tensor
    result = torch.empty((2, count), device=device, dtype=dtype, pin_memory=pin_memory)

    # Compute row offsets on CPU (this is fast since row is typically small)
    row_offsets = _compute_row_offsets(row, col, offset)
    row_offsets_tensor = torch.tensor(row_offsets, dtype=torch.int64, device=device)

    # Use Triton kernel for GPU
    BLOCK_SIZE = 128
    grid = (triton.cdiv(count, BLOCK_SIZE),)

    # Call the Triton kernel
    triu_indices_kernel[grid](
        result, row_offsets_tensor, row, col, offset, count, BLOCK_SIZE
    )

    return result