import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def qkv_projection_fusion_kernel(
    input_ptr,
    output_q_ptr,
    output_k_ptr,
    output_v_ptr,
    n_elements,
    last_dim_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Split tensor of shape [..., 3*hidden_dim] into three [..., hidden_dim] tensors.

    The split is along the last dimension.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Get the position in the last dimension (which is 3*hidden_dim)
    # For input shape [..., 3*hidden_dim], we split into 3 parts of size hidden_dim
    hidden_dim = last_dim_size // 3
    offset_in_last_dim = offsets % last_dim_size

    # Determine which Q/K/V slice this element belongs to
    slice_idx = offset_in_last_dim // hidden_dim

    # Get the offset within the Q/K/V slice
    offset_in_slice = offset_in_last_dim % hidden_dim

    # Compute the base offset in the output (excluding the last dimension)
    # This is offsets // last_dim_size * hidden_dim
    base_out_offset = (offsets // last_dim_size) * hidden_dim

    # Final output offsets
    q_out_offset = base_out_offset + tl.where(slice_idx == 0, offset_in_slice, -1)
    k_out_offset = base_out_offset + tl.where(slice_idx == 1, offset_in_slice, -1)
    v_out_offset = base_out_offset + tl.where(slice_idx == 2, offset_in_slice, -1)

    # Store to respective outputs
    tl.store(output_q_ptr + q_out_offset, input, mask=(slice_idx == 0) & mask)
    tl.store(output_k_ptr + k_out_offset, input, mask=(slice_idx == 1) & mask)
    tl.store(output_v_ptr + v_out_offset, input, mask=(slice_idx == 2) & mask)


def qkv_projection_fusion(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.debug("GEMS QKV_PROJECTION_FUSION")

    assert input.dim() >= 2, "Input must have at least 2 dimensions"
    assert input.shape[-1] % 3 == 0, "Last dimension must be divisible by 3"

    last_dim_size = input.shape[-1]
    n_elements = input.numel()

    # Create output tensors
    output_shape = list(input.shape)
    output_shape[-1] = last_dim_size // 3

    output_q = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    output_k = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    output_v = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    qkv_projection_fusion_kernel[grid](
        input,
        output_q,
        output_k,
        output_v,
        n_elements,
        last_dim_size,
        BLOCK_SIZE,
    )

    return output_q, output_k, output_v