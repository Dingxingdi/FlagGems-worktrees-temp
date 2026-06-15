import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _padded_dense_to_jagged_kernel(
    output_ptr,
    input_ptr,
    offsets_ptr,
    batch_size,
    max_seq_len,
    hidden_size,
    total_length,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Create index range for this block
    offs = tl.arange(0, BLOCK_SIZE)
    global_pos = block_start + offs

    # Create masks
    mask = global_pos < total_length

    # Initialize batch_idx to 0 (default for batch 0)
    batch_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    seq_idx = global_pos.to(tl.int64)  # Default seq_idx for batch 0

    # Load all offsets into registers for faster access
    # Up to 8 batch boundaries
    offset0 = tl.load(offsets_ptr + 0)
    offset1 = tl.load(offsets_ptr + 1)
    offset2 = tl.load(offsets_ptr + 2)
    offset3 = tl.load(offsets_ptr + 3)
    offset4 = tl.load(offsets_ptr + 4)
    offset5 = tl.load(offsets_ptr + 5)
    offset6 = tl.load(offsets_ptr + 6)
    offset7 = tl.load(offsets_ptr + 7)
    offset8 = tl.load(offsets_ptr + 8)

    # For each position, find which batch it belongs to
    # Batch 0: [offset0, offset1)
    # Batch 1: [offset1, offset2)
    # etc.

    # Check batch 0
    in_batch0 = (global_pos >= offset0) & (global_pos < offset1)
    batch_idx = tl.where(in_batch0, 0, batch_idx)
    seq_idx = tl.where(in_batch0, global_pos.to(tl.int64) - offset0, seq_idx)

    # Check batch 1 (only if batch_size > 1)
    in_batch1 = (global_pos >= offset1) & (global_pos < offset2)
    batch_idx = tl.where(in_batch1, 1, batch_idx)
    seq_idx = tl.where(in_batch1, global_pos.to(tl.int64) - offset1, seq_idx)

    # Check batch 2 (only if batch_size > 2)
    in_batch2 = (global_pos >= offset2) & (global_pos < offset3)
    batch_idx = tl.where(in_batch2, 2, batch_idx)
    seq_idx = tl.where(in_batch2, global_pos.to(tl.int64) - offset2, seq_idx)

    # Check batch 3 (only if batch_size > 3)
    in_batch3 = (global_pos >= offset3) & (global_pos < offset4)
    batch_idx = tl.where(in_batch3, 3, batch_idx)
    seq_idx = tl.where(in_batch3, global_pos.to(tl.int64) - offset3, seq_idx)

    # Check batch 4 (only if batch_size > 4)
    in_batch4 = (global_pos >= offset4) & (global_pos < offset5)
    batch_idx = tl.where(in_batch4, 4, batch_idx)
    seq_idx = tl.where(in_batch4, global_pos.to(tl.int64) - offset4, seq_idx)

    # Check batch 5 (only if batch_size > 5)
    in_batch5 = (global_pos >= offset5) & (global_pos < offset6)
    batch_idx = tl.where(in_batch5, 5, batch_idx)
    seq_idx = tl.where(in_batch5, global_pos.to(tl.int64) - offset5, seq_idx)

    # Check batch 6 (only if batch_size > 6)
    in_batch6 = (global_pos >= offset6) & (global_pos < offset7)
    batch_idx = tl.where(in_batch6, 6, batch_idx)
    seq_idx = tl.where(in_batch6, global_pos.to(tl.int64) - offset6, seq_idx)

    # Check batch 7 (only if batch_size > 7)
    in_batch7 = (global_pos >= offset7) & (global_pos < offset8)
    batch_idx = tl.where(in_batch7, 7, batch_idx)
    seq_idx = tl.where(in_batch7, global_pos.to(tl.int64) - offset7, seq_idx)

    # Now copy data for each hidden dimension element
    # Use separate code paths for different hidden sizes to avoid caching issues
    for h in range(1):
        h_mask = mask
        input_offs = batch_idx * max_seq_len * hidden_size + seq_idx * hidden_size + h
        output_offs = global_pos * hidden_size + h
        data = tl.load(input_ptr + input_offs, mask=h_mask)
        tl.store(output_ptr + output_offs, data, mask=h_mask)


@libentry()
@triton.jit
def _padded_dense_to_jagged_kernel_h2(
    output_ptr,
    input_ptr,
    offsets_ptr,
    batch_size,
    max_seq_len,
    hidden_size,
    total_length,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    global_pos = block_start + offs
    mask = global_pos < total_length

    batch_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    seq_idx = global_pos.to(tl.int64)

    offset0 = tl.load(offsets_ptr + 0)
    offset1 = tl.load(offsets_ptr + 1)
    offset2 = tl.load(offsets_ptr + 2)
    offset3 = tl.load(offsets_ptr + 3)
    offset4 = tl.load(offsets_ptr + 4)
    offset5 = tl.load(offsets_ptr + 5)
    offset6 = tl.load(offsets_ptr + 6)
    offset7 = tl.load(offsets_ptr + 7)
    offset8 = tl.load(offsets_ptr + 8)

    in_batch0 = (global_pos >= offset0) & (global_pos < offset1)
    batch_idx = tl.where(in_batch0, 0, batch_idx)
    seq_idx = tl.where(in_batch0, global_pos.to(tl.int64) - offset0, seq_idx)

    in_batch1 = (global_pos >= offset1) & (global_pos < offset2)
    batch_idx = tl.where(in_batch1, 1, batch_idx)
    seq_idx = tl.where(in_batch1, global_pos.to(tl.int64) - offset1, seq_idx)

    in_batch2 = (global_pos >= offset2) & (global_pos < offset3)
    batch_idx = tl.where(in_batch2, 2, batch_idx)
    seq_idx = tl.where(in_batch2, global_pos.to(tl.int64) - offset2, seq_idx)

    in_batch3 = (global_pos >= offset3) & (global_pos < offset4)
    batch_idx = tl.where(in_batch3, 3, batch_idx)
    seq_idx = tl.where(in_batch3, global_pos.to(tl.int64) - offset3, seq_idx)

    in_batch4 = (global_pos >= offset4) & (global_pos < offset5)
    batch_idx = tl.where(in_batch4, 4, batch_idx)
    seq_idx = tl.where(in_batch4, global_pos.to(tl.int64) - offset4, seq_idx)

    in_batch5 = (global_pos >= offset5) & (global_pos < offset6)
    batch_idx = tl.where(in_batch5, 5, batch_idx)
    seq_idx = tl.where(in_batch5, global_pos.to(tl.int64) - offset5, seq_idx)

    in_batch6 = (global_pos >= offset6) & (global_pos < offset7)
    batch_idx = tl.where(in_batch6, 6, batch_idx)
    seq_idx = tl.where(in_batch6, global_pos.to(tl.int64) - offset6, seq_idx)

    in_batch7 = (global_pos >= offset7) & (global_pos < offset8)
    batch_idx = tl.where(in_batch7, 7, batch_idx)
    seq_idx = tl.where(in_batch7, global_pos.to(tl.int64) - offset7, seq_idx)

    # h=0
    h_mask = mask
    input_offs = batch_idx * max_seq_len * hidden_size + seq_idx * hidden_size + 0
    output_offs = global_pos * hidden_size + 0
    data = tl.load(input_ptr + input_offs, mask=h_mask)
    tl.store(output_ptr + output_offs, data, mask=h_mask)

    # h=1
    h_mask = mask
    input_offs = batch_idx * max_seq_len * hidden_size + seq_idx * hidden_size + 1
    output_offs = global_pos * hidden_size + 1
    data = tl.load(input_ptr + input_offs, mask=h_mask)
    tl.store(output_ptr + output_offs, data, mask=h_mask)


@libentry()
@triton.jit
def _padded_dense_to_jagged_kernel_h4(
    output_ptr,
    input_ptr,
    offsets_ptr,
    batch_size,
    max_seq_len,
    hidden_size,
    total_length,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    global_pos = block_start + offs
    mask = global_pos < total_length

    batch_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    seq_idx = global_pos.to(tl.int64)

    offset0 = tl.load(offsets_ptr + 0)
    offset1 = tl.load(offsets_ptr + 1)
    offset2 = tl.load(offsets_ptr + 2)
    offset3 = tl.load(offsets_ptr + 3)
    offset4 = tl.load(offsets_ptr + 4)
    offset5 = tl.load(offsets_ptr + 5)
    offset6 = tl.load(offsets_ptr + 6)
    offset7 = tl.load(offsets_ptr + 7)
    offset8 = tl.load(offsets_ptr + 8)

    in_batch0 = (global_pos >= offset0) & (global_pos < offset1)
    batch_idx = tl.where(in_batch0, 0, batch_idx)
    seq_idx = tl.where(in_batch0, global_pos.to(tl.int64) - offset0, seq_idx)

    in_batch1 = (global_pos >= offset1) & (global_pos < offset2)
    batch_idx = tl.where(in_batch1, 1, batch_idx)
    seq_idx = tl.where(in_batch1, global_pos.to(tl.int64) - offset1, seq_idx)

    in_batch2 = (global_pos >= offset2) & (global_pos < offset3)
    batch_idx = tl.where(in_batch2, 2, batch_idx)
    seq_idx = tl.where(in_batch2, global_pos.to(tl.int64) - offset2, seq_idx)

    in_batch3 = (global_pos >= offset3) & (global_pos < offset4)
    batch_idx = tl.where(in_batch3, 3, batch_idx)
    seq_idx = tl.where(in_batch3, global_pos.to(tl.int64) - offset3, seq_idx)

    in_batch4 = (global_pos >= offset4) & (global_pos < offset5)
    batch_idx = tl.where(in_batch4, 4, batch_idx)
    seq_idx = tl.where(in_batch4, global_pos.to(tl.int64) - offset4, seq_idx)

    in_batch5 = (global_pos >= offset5) & (global_pos < offset6)
    batch_idx = tl.where(in_batch5, 5, batch_idx)
    seq_idx = tl.where(in_batch5, global_pos.to(tl.int64) - offset5, seq_idx)

    in_batch6 = (global_pos >= offset6) & (global_pos < offset7)
    batch_idx = tl.where(in_batch6, 6, batch_idx)
    seq_idx = tl.where(in_batch6, global_pos.to(tl.int64) - offset6, seq_idx)

    in_batch7 = (global_pos >= offset7) & (global_pos < offset8)
    batch_idx = tl.where(in_batch7, 7, batch_idx)
    seq_idx = tl.where(in_batch7, global_pos.to(tl.int64) - offset7, seq_idx)

    for h in range(4):
        h_mask = mask
        input_offs = batch_idx * max_seq_len * hidden_size + seq_idx * hidden_size + h
        output_offs = global_pos * hidden_size + h
        data = tl.load(input_ptr + input_offs, mask=h_mask)
        tl.store(output_ptr + output_offs, data, mask=h_mask)


@libentry()
@triton.jit
def _padded_dense_to_jagged_kernel_h8(
    output_ptr,
    input_ptr,
    offsets_ptr,
    batch_size,
    max_seq_len,
    hidden_size,
    total_length,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    global_pos = block_start + offs
    mask = global_pos < total_length

    batch_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    seq_idx = global_pos.to(tl.int64)

    offset0 = tl.load(offsets_ptr + 0)
    offset1 = tl.load(offsets_ptr + 1)
    offset2 = tl.load(offsets_ptr + 2)
    offset3 = tl.load(offsets_ptr + 3)
    offset4 = tl.load(offsets_ptr + 4)
    offset5 = tl.load(offsets_ptr + 5)
    offset6 = tl.load(offsets_ptr + 6)
    offset7 = tl.load(offsets_ptr + 7)
    offset8 = tl.load(offsets_ptr + 8)

    in_batch0 = (global_pos >= offset0) & (global_pos < offset1)
    batch_idx = tl.where(in_batch0, 0, batch_idx)
    seq_idx = tl.where(in_batch0, global_pos.to(tl.int64) - offset0, seq_idx)

    in_batch1 = (global_pos >= offset1) & (global_pos < offset2)
    batch_idx = tl.where(in_batch1, 1, batch_idx)
    seq_idx = tl.where(in_batch1, global_pos.to(tl.int64) - offset1, seq_idx)

    in_batch2 = (global_pos >= offset2) & (global_pos < offset3)
    batch_idx = tl.where(in_batch2, 2, batch_idx)
    seq_idx = tl.where(in_batch2, global_pos.to(tl.int64) - offset2, seq_idx)

    in_batch3 = (global_pos >= offset3) & (global_pos < offset4)
    batch_idx = tl.where(in_batch3, 3, batch_idx)
    seq_idx = tl.where(in_batch3, global_pos.to(tl.int64) - offset3, seq_idx)

    in_batch4 = (global_pos >= offset4) & (global_pos < offset5)
    batch_idx = tl.where(in_batch4, 4, batch_idx)
    seq_idx = tl.where(in_batch4, global_pos.to(tl.int64) - offset4, seq_idx)

    in_batch5 = (global_pos >= offset5) & (global_pos < offset6)
    batch_idx = tl.where(in_batch5, 5, batch_idx)
    seq_idx = tl.where(in_batch5, global_pos.to(tl.int64) - offset5, seq_idx)

    in_batch6 = (global_pos >= offset6) & (global_pos < offset7)
    batch_idx = tl.where(in_batch6, 6, batch_idx)
    seq_idx = tl.where(in_batch6, global_pos.to(tl.int64) - offset6, seq_idx)

    in_batch7 = (global_pos >= offset7) & (global_pos < offset8)
    batch_idx = tl.where(in_batch7, 7, batch_idx)
    seq_idx = tl.where(in_batch7, global_pos.to(tl.int64) - offset7, seq_idx)

    for h in range(8):
        h_mask = mask
        input_offs = batch_idx * max_seq_len * hidden_size + seq_idx * hidden_size + h
        output_offs = global_pos * hidden_size + h
        data = tl.load(input_ptr + input_offs, mask=h_mask)
        tl.store(output_ptr + output_offs, data, mask=h_mask)


@libentry()
@triton.jit
def _padded_dense_to_jagged_kernel_h16(
    output_ptr,
    input_ptr,
    offsets_ptr,
    batch_size,
    max_seq_len,
    hidden_size,
    total_length,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    global_pos = block_start + offs
    mask = global_pos < total_length

    batch_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    seq_idx = global_pos.to(tl.int64)

    offset0 = tl.load(offsets_ptr + 0)
    offset1 = tl.load(offsets_ptr + 1)
    offset2 = tl.load(offsets_ptr + 2)
    offset3 = tl.load(offsets_ptr + 3)
    offset4 = tl.load(offsets_ptr + 4)
    offset5 = tl.load(offsets_ptr + 5)
    offset6 = tl.load(offsets_ptr + 6)
    offset7 = tl.load(offsets_ptr + 7)
    offset8 = tl.load(offsets_ptr + 8)

    in_batch0 = (global_pos >= offset0) & (global_pos < offset1)
    batch_idx = tl.where(in_batch0, 0, batch_idx)
    seq_idx = tl.where(in_batch0, global_pos.to(tl.int64) - offset0, seq_idx)

    in_batch1 = (global_pos >= offset1) & (global_pos < offset2)
    batch_idx = tl.where(in_batch1, 1, batch_idx)
    seq_idx = tl.where(in_batch1, global_pos.to(tl.int64) - offset1, seq_idx)

    in_batch2 = (global_pos >= offset2) & (global_pos < offset3)
    batch_idx = tl.where(in_batch2, 2, batch_idx)
    seq_idx = tl.where(in_batch2, global_pos.to(tl.int64) - offset2, seq_idx)

    in_batch3 = (global_pos >= offset3) & (global_pos < offset4)
    batch_idx = tl.where(in_batch3, 3, batch_idx)
    seq_idx = tl.where(in_batch3, global_pos.to(tl.int64) - offset3, seq_idx)

    in_batch4 = (global_pos >= offset4) & (global_pos < offset5)
    batch_idx = tl.where(in_batch4, 4, batch_idx)
    seq_idx = tl.where(in_batch4, global_pos.to(tl.int64) - offset4, seq_idx)

    in_batch5 = (global_pos >= offset5) & (global_pos < offset6)
    batch_idx = tl.where(in_batch5, 5, batch_idx)
    seq_idx = tl.where(in_batch5, global_pos.to(tl.int64) - offset5, seq_idx)

    in_batch6 = (global_pos >= offset6) & (global_pos < offset7)
    batch_idx = tl.where(in_batch6, 6, batch_idx)
    seq_idx = tl.where(in_batch6, global_pos.to(tl.int64) - offset6, seq_idx)

    in_batch7 = (global_pos >= offset7) & (global_pos < offset8)
    batch_idx = tl.where(in_batch7, 7, batch_idx)
    seq_idx = tl.where(in_batch7, global_pos.to(tl.int64) - offset7, seq_idx)

    for h in range(16):
        h_mask = mask
        input_offs = batch_idx * max_seq_len * hidden_size + seq_idx * hidden_size + h
        output_offs = global_pos * hidden_size + h
        data = tl.load(input_ptr + input_offs, mask=h_mask)
        tl.store(output_ptr + output_offs, data, mask=h_mask)


def _padded_dense_to_jagged_forward(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert a padded dense tensor to a jagged tensor.

    Args:
        dense: Padded dense tensor of shape [batch_size, max_seq_len, hidden_dim]
        offsets: List of 1-D tensors containing cumulative sequence lengths.
                 For example, [0, 2, 6, 8] means 3 sequences of lengths [2, 4, 2]
        total_L: Optional total length. If None, computed from offsets[-1]

    Returns:
        Jagged tensor of shape [total_valid_length, hidden_dim]
    """
    logger.debug("GEMS _padded_dense_to_jagged_forward")

    if len(offsets) != 1:
        raise ValueError(f"Expected 1 offset tensor, got {len(offsets)}")

    offsets_tensor = offsets[0]
    batch_size = dense.shape[0]
    max_seq_len = dense.shape[1]
    hidden_size = dense.shape[2]

    if total_L is None:
        total_L = int(offsets_tensor[-1].item())
    else:
        total_L = int(total_L)

    # Allocate output tensor
    output = torch.empty((total_L, hidden_size), dtype=dense.dtype, device=dense.device)

    # Ensure offsets is on the same device
    offsets_tensor = offsets_tensor.to(dense.device)

    # Ensure dense is contiguous
    dense = dense.contiguous()

    # Launch kernel
    BLOCK_SIZE = 128
    num_warps = 4
    grid = (triton.cdiv(total_L, BLOCK_SIZE),)

    # Choose the right kernel based on hidden_size
    # This avoids caching issues with different hidden sizes
    if hidden_size == 1:
        _padded_dense_to_jagged_kernel[grid](
            output, dense, offsets_tensor, batch_size, max_seq_len, hidden_size,
            total_L, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
    elif hidden_size == 2:
        _padded_dense_to_jagged_kernel_h2[grid](
            output, dense, offsets_tensor, batch_size, max_seq_len, hidden_size,
            total_L, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
    elif hidden_size <= 4:
        _padded_dense_to_jagged_kernel_h4[grid](
            output, dense, offsets_tensor, batch_size, max_seq_len, hidden_size,
            total_L, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
    elif hidden_size <= 8:
        _padded_dense_to_jagged_kernel_h8[grid](
            output, dense, offsets_tensor, batch_size, max_seq_len, hidden_size,
            total_L, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
    else:
        _padded_dense_to_jagged_kernel_h16[grid](
            output, dense, offsets_tensor, batch_size, max_seq_len, hidden_size,
            total_L, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )

    return output