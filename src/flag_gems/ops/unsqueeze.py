import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _unsqueeze_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def _unsqueeze_impl(input: torch.Tensor, dim: int):
    """Implementation that uses a Triton kernel for data movement."""
    # Calculate output shape
    dim = dim if dim >= 0 else input.dim() + dim + 1
    out_shape = list(input.shape)
    out_shape.insert(dim, 1)

    # Create output tensor
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Ensure contiguous memory for efficient linear copy
    src = input.contiguous() if not input.is_contiguous() else input
    if not out.is_contiguous():
        out = out.contiguous()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        _unsqueeze_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
    return out


def unsqueeze(input: torch.Tensor, dim: int) -> torch.Tensor:
    logger.debug("GEMS UNSQUEEZE")
    """
    Returns a new tensor with a dimension of size one inserted at the
    specified position.

    Args:
        input (Tensor): the input tensor.
        dim (int): the index at which to insert the singleton dimension

    Returns:
        Tensor: the output tensor
    """
    return _unsqueeze_impl(input, dim)


def unsqueeze_(input: torch.Tensor, dim: int) -> torch.Tensor:
    logger.debug("GEMS UNSQUEEZE_")
    """
    In-place version of unsqueeze. Inserts a dimension of size one at the
    specified position and returns the same tensor.

    Args:
        input (Tensor): the input tensor.
        dim (int): the index at which to insert the singleton dimension

    Returns:
        Tensor: the same tensor with the singleton dimension inserted
    """
    # Save the original data
    original_data = input.clone()
    # Calculate output shape
    dim = dim if dim >= 0 else input.dim() + dim + 1
    out_shape = list(input.shape)
    out_shape.insert(dim, 1)
    # Resize the input tensor
    input.resize_(out_shape)
    # Copy the data back
    result = _unsqueeze_impl(original_data, dim)
    input.copy_(result)
    return input


def unsqueeze_copy(input: torch.Tensor, dim: int) -> torch.Tensor:
    logger.debug("GEMS UNSQUEEZE_COPY")
    """
    Returns a copy of the input tensor with a dimension of size one inserted
    at the specified position.

    Args:
        input (Tensor): the input tensor.
        dim (int): the index at which to insert the singleton dimension

    Returns:
        Tensor: a copy of the input tensor with the singleton dimension inserted
    """
    return _unsqueeze_impl(input, dim)