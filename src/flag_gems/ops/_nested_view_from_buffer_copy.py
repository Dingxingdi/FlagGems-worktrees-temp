import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _nested_view_from_buffer_copy_kernel(
    output_ptr,
    input_ptr,
    offsets_ptr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the global position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_offs = tl.arange(0, BLOCK_SIZE)

    # Load mask
    mask = block_offs < num_elements

    # Load values from input buffer at offsets
    # Each thread loads one element from the correct position in the buffer
    input_offsets = tl.load(offsets_ptr + block_offs, mask=mask, other=0)
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0)

    # Store to output
    tl.store(output_ptr + block_start + block_offs, values, mask=mask)


def _nested_view_from_buffer_copy(
    buffer: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Create a nested tensor from a flat buffer.

    This implementation uses Triton to copy data from the flat buffer,
    then uses PyTorch's jagged layout to create the nested tensor.

    Args:
        buffer: Flat input tensor
        nested_size: Shape [num_tensors, max_dims] - sizes of each sub-tensor
        nested_strides: Shape [num_tensors, max_dims] - strides for each sub-tensor
        offsets: Shape [num_tensors] - starting position in buffer for each sub-tensor

    Returns:
        A nested tensor
    """
    logger.debug("GEMS _nested_view_from_buffer_copy")

    # Get number of tensors
    num_tensors = nested_size.shape[0]

    # Calculate total output size (sum of all sub-tensor sizes)
    total_size = nested_size.sum().item()

    if total_size == 0:
        # Empty case - return empty nested tensor
        return torch.nested.nested_tensor([], layout=torch.jagged, device=buffer.device)

    # Create offsets for each element in the output
    # These tell us where each element comes from in the input buffer
    # For each sub-tensor i, elements j=0..size[i]-1 come from buffer at offsets[i] + j*stride[i]
    sizes = nested_size.squeeze(-1)  # [num_tensors]
    strides = nested_strides.squeeze(-1)  # [num_tensors]

    # Build element offsets: for each element in the output, where does it come from?
    element_offsets = []
    for i in range(num_tensors):
        size_i = sizes[i].item()
        stride_i = strides[i].item()
        offset_i = offsets[i].item()
        for j in range(size_i):
            element_offsets.append(offset_i + j * stride_i)

    element_offsets = torch.tensor(
        element_offsets, dtype=torch.long, device=buffer.device
    )

    # Use Triton kernel to copy data
    output = torch.empty(total_size, dtype=buffer.dtype, device=buffer.device)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(total_size, BLOCK_SIZE),)

    _nested_view_from_buffer_copy_kernel[grid](
        output,
        buffer,
        element_offsets,
        total_size,
        BLOCK_SIZE,
    )

    # Now create the nested tensor using jagged layout
    # We need to compute the offsets for jagged layout (cumulative sizes)
    jagged_offsets = torch.zeros(num_tensors + 1, dtype=torch.long, device=buffer.device)
    jagged_offsets[1:] = torch.cumsum(sizes, dim=0)

    result = torch.nested.nested_tensor_from_jagged(output, offsets=jagged_offsets)

    return result


def nested_view_from_buffer_copy(
    buffer: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
):
    """Public wrapper for _nested_view_from_buffer_copy"""
    return _nested_view_from_buffer_copy(buffer, nested_size, nested_strides, offsets)