import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def jagged_to_padded_dense_forward_kernel(
    values_ptr,
    offsets_ptr,
    out_ptr,
    padding_value,  # Don't use tl.constexpr for float types
    batch_size: tl.constexpr,
    max_length: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Kernel for converting jagged (variable-length) sequences to padded dense tensor.

    Args:
        values: Flat tensor of shape [total_values, feature_size] containing all sequence values
        offsets: Tensor of shape [batch_size + 1] containing cumulative offsets
        out: Output tensor of shape [batch_size, max_length, feature_size]
        padding_value: Value to use for padding
    """
    pid = tle.program_id(0)

    # Each program handles one sequence
    if pid >= batch_size:
        return

    seq_idx = pid
    seq_start_offset = tl.load(offsets_ptr + seq_idx)
    seq_length = tl.load(offsets_ptr + seq_idx + 1) - seq_start_offset

    # Iterate over positions in the sequence
    for pos in range(max_length):
        # Calculate the source index in values
        src_idx = seq_start_offset + pos

        # Determine if this position is valid (within sequence length)
        is_valid = pos < seq_length

        # Iterate over feature dimensions
        for feat in range(feature_size):
            # Output position
            out_pos = seq_idx * max_length * feature_size + pos * feature_size + feat

            if is_valid:
                # Load from values tensor
                value_pos = src_idx * feature_size + feat
                val = tl.load(values_ptr + value_pos)
            else:
                # Padding - use 0 cast to output dtype (via loaded value type)
                val = padding_value

            tl.store(out_ptr + out_pos, val)


def _jagged_to_padded_dense_forward(values, offsets, max_lengths, padding_value=0.0):
    logger.debug("GEMS JAGGED TO PADDED DENSE FORWARD")

    # Ensure offsets is a list of tensors
    if isinstance(offsets, (list, tuple)):
        offsets = offsets[0]

    batch_size = offsets.numel() - 1  # offsets has batch_size + 1 elements
    max_length = max_lengths[0] if isinstance(max_lengths, (list, tuple)) else max_lengths

    # Handle feature dimension
    if values.ndim == 1:
        feature_size = 1
        total_values = values.numel()
        values = values.view(-1, 1)  # Reshape to 2D for kernel
    else:
        feature_size = values.shape[1] if values.ndim > 1 else 1
        total_values = values.numel() // feature_size

    # Create output tensor - match PyTorch behavior:
    # - 1D input -> 2D output [batch_size, max_length]
    # - 2D+ input -> 3D+ output [batch_size, max_length, feature_size]
    if feature_size == 1:
        out_shape = (batch_size, max_length)  # No feature dimension for 1D input
    else:
        out_shape = (batch_size, max_length, feature_size)

    # For float16/bfloat16, compute in float32 and convert back
    orig_dtype = values.dtype
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.to(torch.float32)

    out = torch.full(out_shape, padding_value, dtype=values.dtype, device=values.device)

    # Launch kernel - one program per sequence
    BLOCK_M = 32
    grid = (batch_size,)

    jagged_to_padded_dense_forward_kernel[grid](
        values,
        offsets,
        out,
        padding_value,
        batch_size,
        max_length,
        feature_size,
        BLOCK_M,
    )

    # Convert back to original dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        out = out.to(orig_dtype)

    return out