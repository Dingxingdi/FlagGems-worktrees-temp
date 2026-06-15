import logging
from typing import List

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def unsafe_split_offset_kernel(
    out_ptr,
    in_ptr,
    total_elements,
    out_dim_size,
    dim_prod_post,
    in_dim_size,  # Original input size in the split dimension
    dim_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy a slice of input tensor to output tensor for unsafe_split.

    This kernel copies data from input to output, starting at a specific offset
    along the split dimension.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets

    # Compute indices in the output tensor
    # output is organized as: [pre][out_dim][post]
    pre_idx = idx // (out_dim_size * dim_prod_post)
    out_dim_idx = (idx // dim_prod_post) % out_dim_size
    post_idx = idx % dim_prod_post

    # Compute the corresponding input index
    # The input index formula:
    # - pre_idx * in_dim_size * dim_prod_post: skip complete "pre" blocks
    # - (out_dim_idx + dim_offset) * dim_prod_post: offset into the dimension
    # - post_idx: post dimension offset
    in_idx = (
        pre_idx * in_dim_size * dim_prod_post
        + (out_dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    # Load from input and store to output
    data = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + idx, data, mask=mask)


def unsafe_split(
    tensor: torch.Tensor, split_size: int, dim: int = 0
) -> List[torch.Tensor]:
    """Split a tensor into chunks of size split_size along dimension dim.

    This implementation uses Triton kernel to copy data to each output tensor.
    While PyTorch's unsafe_split returns views, this implementation creates
    independent tensors for compatibility with the Triton kernel approach.
    """
    logger.debug("GEMS UNSAFE_SPLIT")

    assert dim >= -tensor.ndim and dim < tensor.ndim, "Dimension out of range"
    dim = dim % tensor.ndim

    if split_size <= 0:
        raise ValueError("split_size must be positive")

    # For non-contiguous tensors, make them contiguous for simpler indexing
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    dim_size = tensor.shape[dim]
    dim_prod_post = 1
    for d in range(dim + 1, tensor.ndim):
        dim_prod_post *= tensor.shape[d]

    results: List[torch.Tensor] = []
    dim_offset = 0

    while dim_offset < dim_size:
        current_split_size = min(split_size, dim_size - dim_offset)

        out_shape = list(tensor.shape)
        out_shape[dim] = current_split_size

        out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)

        total_elements = out.numel()

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

        unsafe_split_offset_kernel[grid](
            out,
            tensor,
            total_elements,
            current_split_size,
            dim_prod_post,
            dim_size,
            dim_offset,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        results.append(out)
        dim_offset += current_split_size

    return results