import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def split_copy_kernel(
    out_ptr,
    in_ptr,
    dim_size_in,
    dim_size_out,
    dim_prod_post,
    dim_prod_pre,
    dim_offset,
    total_elements,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)

    block_start = pid_x * BLOCK_X
    offsets = tl.arange(0, BLOCK_X)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets

    # Compute coordinates in output tensor
    # Output shape: (..., dim_size_out, ...)
    # The dimension being split is at position determined by dim_prod_pre and dim_prod_post
    pre_idx = idx // (dim_size_out * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size_out
    post_idx = idx % dim_prod_post

    # Map to input coordinates: add offset to the split dimension
    in_dim_idx = dim_idx + dim_offset

    # Compute flat index in input tensor
    in_idx = pre_idx * dim_size_in * dim_prod_post + in_dim_idx * dim_prod_post + post_idx

    data = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + idx, data, mask=mask)


def split(
    self: torch.Tensor,
    split_size_or_sections: Union[int, List[int]],
    dim: int = 0,
) -> Tuple[torch.Tensor, ...]:
    logger.debug("GEMS SPLIT")
    # Validate input tensor
    if self.ndim == 0:
        raise RuntimeError("split() requires a tensor with at least 1 dimension")

    # Normalize dim
    dim = dim % self.ndim

    # Handle integer split_size
    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        if split_size <= 0:
            raise RuntimeError("split_size must be greater than 0")

        dim_size = self.shape[dim]
        # Calculate the number of splits
        num_splits = (dim_size + split_size - 1) // split_size

        # Generate split sizes
        split_sizes = []
        for i in range(num_splits):
            start = i * split_size
            end = min(start + split_size, dim_size)
            split_sizes.append(end - start)
    else:
        # Handle list of split sizes
        split_sizes = list(split_size_or_sections)
        if len(split_sizes) == 0:
            raise RuntimeError("split_size_or_sections must not be empty")

        # Validate split sizes sum to the dimension size
        dim_size = self.shape[dim]
        if sum(split_sizes) != dim_size:
            raise RuntimeError(
                f"split_size_or_sections ({split_sizes}) must sum to "
                f"the dimension size ({dim_size})"
            )

    # Build output shapes
    output_shapes = []
    for size in split_sizes:
        shape = list(self.shape)
        shape[dim] = size
        output_shapes.append(shape)

    # Create output tensors
    outputs = []
    for shape in output_shapes:
        outputs.append(torch.empty(shape, dtype=self.dtype, device=self.device))

    # Calculate stride info for the split dimension
    dim_prod_post = 1
    for d in range(dim + 1, self.ndim):
        dim_prod_post *= self.shape[d]

    dim_prod_pre = 1
    for d in range(dim):
        dim_prod_pre *= self.shape[d]

    # If there's only one output, just return a copy
    if len(outputs) == 1:
        outputs[0].copy_(self)
        return tuple(outputs)

    # Copy each split to its output
    dim_offset = 0
    BLOCK = 1024

    for i, out_tensor in enumerate(outputs):
        dim_size_out = split_sizes[i]
        if dim_size_out == 0:
            dim_offset += dim_size_out
            continue

        total_elements = out_tensor.numel()

        grid = (triton.cdiv(total_elements, BLOCK),)

        split_copy_kernel[grid](
            out_tensor,
            self,
            self.shape[dim],
            dim_size_out,
            dim_prod_post,
            dim_prod_pre,
            dim_offset,
            total_elements,
            BLOCK_X=BLOCK,
        )

        dim_offset += dim_size_out

    return tuple(outputs)


# Also implement split_with_sizes which is the aten version
def split_with_sizes(
    self: torch.Tensor,
    split_sizes: List[int],
    dim: int = 0,
) -> Tuple[torch.Tensor, ...]:
    logger.debug("GEMS SPLIT_WITH_SIZES")
    return split(self, split_sizes, dim)