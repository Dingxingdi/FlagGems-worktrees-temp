import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _resize_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, x, mask=mask)


def resize(A: torch.Tensor, size, memory_format=None) -> torch.Tensor:
    """Resize tensor to the specified size.

    This is a functional version that returns a new tensor.
    PyTorch's resize semantics: data is reshaped in row-major (C) order.
    """
    logger.debug("GEMS RESIZE")

    # Convert size to tuple if it's a list
    if not isinstance(size, (tuple, torch.Size)):
        size = tuple(size)

    # If size is the same, return the original tensor
    if tuple(A.shape) == size:
        return A

    # Get the minimum number of elements
    new_numel = 1
    for dim in size:
        new_numel *= dim
    old_numel = A.numel()

    if old_numel == new_numel:
        # Same number of elements, just reshape
        return A.reshape(size)

    # Different number of elements - need to copy data
    # First, reshape to 1D to get the flattened data in row-major order
    A_flat = A.reshape(-1)
    out = torch.empty(size, dtype=A.dtype, device=A.device)

    # Copy data - min of old and new numel
    n_elements = min(old_numel, new_numel)

    if n_elements > 0:
        # Copy first n_elements from flat input to output
        # Using Triton kernel for the copy
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _resize_copy_kernel[grid](A_flat, out, n_elements, BLOCK_SIZE=1024)

    return out


def resize_(A: torch.Tensor, size, memory_format=None) -> torch.Tensor:
    """In-place resize tensor to the specified size.

    This modifies the input tensor in-place.
    PyTorch's resize semantics: data is reshaped in row-major (C) order.
    """
    logger.debug("GEMS RESIZE_")

    # Convert size to tuple if it's a list
    if not isinstance(size, (tuple, torch.Size)):
        size = tuple(size)

    # If size is the same, return the original tensor
    if tuple(A.shape) == size:
        return A

    # Get the minimum number of elements
    new_numel = 1
    for dim in size:
        new_numel *= dim
    old_numel = A.numel()

    if old_numel == new_numel:
        # Same number of elements, reshape in-place
        A.resize_(size)
        return A

    # Different number of elements
    # Use resize to create new tensor with new size
    new_tensor = A.resize_(size)

    # Copy data using Triton kernel for the overlap
    n_elements = min(old_numel, new_numel)
    if n_elements > 0:
        original_flat = A.reshape(-1).contiguous()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _resize_copy_kernel[grid](
            original_flat, new_tensor.contiguous(), n_elements, BLOCK_SIZE=1024
        )

    return A