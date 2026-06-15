import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# Pointwise kernel for copying data with reshape
# This is used when the input cannot be viewed with the new shape
@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _reshape_copy_kernel(x):
    return x


def _try_view(inp, shape):
    """
    Try to create a view of the input tensor with the new shape.
    Returns the view if possible, None otherwise.
    """
    try:
        return inp.view(shape)
    except RuntimeError:
        return None


def reshape(inp, shape) -> torch.Tensor:
    """
    Reshape the input tensor to the given shape.

    If possible, returns a view of the input tensor without copying data.
    Otherwise, copies the data to a new tensor with the new shape.
    """
    logger.debug("GEMS RESHAPE")

    # Convert shape to tuple if it's not already
    if not isinstance(shape, (tuple, list)):
        shape = tuple(shape) if hasattr(shape, '__iter__') else (shape,)

    # Try view first - most efficient if possible (no copy)
    result = _try_view(inp, shape)
    if result is not None:
        return result

    # If view is not possible, we need to copy the data
    # Use pointwise_dynamic kernel to do the copy
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    overload = _reshape_copy_kernel.instantiate(inp.ndim)
    overload(inp, out0=out)
    return out