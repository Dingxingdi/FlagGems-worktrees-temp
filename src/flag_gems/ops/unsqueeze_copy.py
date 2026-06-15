import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


def unsqueeze_copy(inp: torch.Tensor, dim: int) -> torch.Tensor:
    logger.debug("GEMS UNSQUEEZE_COPY")

    # Normalize dim
    ndim = inp.dim()
    if dim < 0:
        dim = dim + ndim + 1

    # Build output shape
    shape = list(inp.shape)
    shape.insert(dim, 1)

    # Create output tensor
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)

    # Reshape input to match output shape for broadcasting
    # After unsqueeze, the input data should be at positions where dim index is removed
    # e.g., input shape (2, 3), dim=1 -> output shape (2, 1, 3)
    # We need to copy input[*, *] to output[*, 0, *]
    # This can be done by reshaping input to (2, 1, 3) and then copying

    # Create a view of input that matches output shape
    inp_view = inp.view(shape)

    # Use copy kernel to copy data
    overload = _copy_kernel.instantiate(len(shape))
    overload(inp_view, out0=out)

    return out


def unsqueeze_copy_(inp: torch.Tensor, dim: int) -> torch.Tensor:
    """In-place version of unsqueeze_copy."""
    logger.debug("GEMS UNSQUEEZE_COPY_")

    # Normalize dim
    ndim = inp.dim()
    if dim < 0:
        dim = dim + ndim + 1

    # Build output shape
    shape = list(inp.shape)
    shape.insert(dim, 1)

    # Save original data
    original_data = inp.clone()

    # Resize the tensor in-place
    inp.resize_(shape)

    # Copy original data to the new tensor using broadcast
    # Create a view of original data with the new shape
    original_view = original_data.view(inp.shape)

    # Copy data back to inp
    overload = _copy_kernel.instantiate(len(shape))
    overload(original_view, out0=inp)

    return inp