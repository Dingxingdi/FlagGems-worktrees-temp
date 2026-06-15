import logging
from typing import List

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def pad_sequence_kernel(
    output_ptr,
    seq_data_ptr,  # Pointer to concatenated sequence data
    seq_offsets_ptr,  # Starting offset for each sequence in the concatenated data
    seq_lengths_ptr,  # Length of each sequence
    max_seq_len,
    num_seqs,
    padding_value,
    padding_side_is_right: tl.constexpr,
    feature_size,
    batch_first: tl.constexpr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to copy sequence data to output tensor.

    Args:
        output_ptr: Pointer to output tensor
        seq_data_ptr: Pointer to concatenated sequence data
        seq_offsets_ptr: Starting offset for each sequence in concatenated data
        seq_lengths_ptr: Lengths of each sequence
        max_seq_len: Maximum sequence length
        num_seqs: Number of sequences
        padding_value: Padding value (float)
        padding_side_is_right: 1 for right padding, 0 for left
        feature_size: Feature dimension size
        batch_first: 1 for batch_first=True, 0 for batch_first=False
        total_elements: Total elements in output
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets

    # Output layout: [num_seqs, max_seq_len, feature_size] if batch_first
    #                 [max_seq_len, num_seqs, feature_size] if not batch_first
    if batch_first:
        seq_idx = idx // (max_seq_len * feature_size)
        remainder = idx % (max_seq_len * feature_size)
        seq_pos = remainder // feature_size
        feature_idx = remainder % feature_size
    else:
        seq_pos = idx // (num_seqs * feature_size)
        remainder = idx % (num_seqs * feature_size)
        seq_idx = remainder // feature_size
        feature_idx = remainder % feature_size

    # Load sequence length for this sequence
    length = tl.load(seq_lengths_ptr + seq_idx)

    # Determine if padding position
    if padding_side_is_right:
        is_padding = seq_pos >= length
    else:
        is_padding = seq_pos < (max_seq_len - length)

    # Initialize with padding value for all positions
    result = tl.full((BLOCK_SIZE,), padding_value, tl.float32)

    # For valid (non-padding) positions, load from source data
    valid_mask = ~is_padding & mask

    if padding_side_is_right:
        src_seq_pos = seq_pos
    else:
        src_seq_pos = seq_pos - (max_seq_len - length)

    # Compute source offset
    seq_offset = tl.load(seq_offsets_ptr + seq_idx)
    src_offset = seq_offset + src_seq_pos * feature_size + feature_idx

    # Load from source for valid positions
    valid_data = tl.load(seq_data_ptr + src_offset, mask=valid_mask)

    # Use valid data where available, otherwise keep padding value
    result = tl.where(valid_mask, valid_data, result)

    # Store result
    tl.store(output_ptr + idx, result, mask=mask)


def pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
) -> torch.Tensor:
    """Pad a list of variable length Tensors with padding_value.

    Args:
        sequences: List of variable length sequences, each of shape [L, *] where L is
                   the length of the sequence and * is any number of dimensions.
        batch_first: If True, output will be in [B, T, *] format, otherwise [T, B, *].
        padding_value: Value to use for padded elements. Default: 0.0.
        padding_side: Side to pad on, "right" or "left". Default: "right".

    Returns:
        Tensor of padded and stacked sequences.
    """
    logger.debug("GEMS PAD_SEQUENCE")

    if len(sequences) == 0:
        raise ValueError("pad_sequence(): expected a non-empty list of Tensors")

    if len(sequences) == 1:
        # Single sequence - just return it reshaped appropriately
        seq = sequences[0]
        if batch_first:
            return seq.unsqueeze(0)
        else:
            return seq.unsqueeze(1)

    # Validate all sequences have the same dtype and device
    device = sequences[0].device
    dtype = sequences[0].dtype

    for seq in sequences:
        if seq.device != device:
            raise ValueError("All sequences must be on the same device")
        if seq.dtype != dtype:
            raise ValueError("All sequences must have the same dtype")

    # Compute sequence lengths
    seq_lengths = [s.shape[0] for s in sequences]
    max_seq_len = max(seq_lengths)
    num_seqs = len(sequences)

    # Get feature shape from first sequence (all must have same feature dims)
    if sequences[0].ndim == 1:
        feature_shape = [1]
    else:
        feature_shape = list(sequences[0].shape[1:])

    feature_size = 1
    for dim_size in feature_shape:
        feature_size *= dim_size

    # Create output tensor
    if batch_first:
        out_shape = [num_seqs, max_seq_len] + feature_shape
    else:
        out_shape = [max_seq_len, num_seqs] + feature_shape

    output = torch.full(out_shape, padding_value, dtype=dtype, device=device)
    total_elements = output.numel()

    # Flatten all sequences into a single contiguous tensor
    flat_sequences = []
    for seq in sequences:
        flat_sequences.append(seq.reshape(-1))
    flat_data = torch.cat(flat_sequences)

    # Compute offsets for each sequence in the flattened data
    offsets_list = [0]
    offset = 0
    for seq in sequences:
        offset += seq.numel()
        offsets_list.append(offset)
    seq_offsets_tensor = torch.tensor(offsets_list[:-1], device=device, dtype=torch.int64)

    # Convert lengths to tensor
    seq_lengths_tensor = torch.tensor(seq_lengths, device=device, dtype=torch.int32)

    # Launch Triton kernel
    BLOCK_SIZE = 128
    padding_side_is_right = 1 if padding_side == "right" else 0
    batch_first_flag = 1 if batch_first else 0

    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    pad_sequence_kernel[grid](
        output,
        flat_data,
        seq_offsets_tensor,
        seq_lengths_tensor,
        max_seq_len,
        num_seqs,
        padding_value,
        padding_side_is_right,
        feature_size,
        batch_first_flag,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# For compatibility with aten registration
def _pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
) -> torch.Tensor:
    return pad_sequence(sequences, batch_first, padding_value, padding_side)