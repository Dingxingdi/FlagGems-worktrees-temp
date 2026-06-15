import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _squeeze_identity(x):
    """Identity function for squeeze - just returns the input value."""
    return x


def squeeze(inp, dim=None):
    """
    Wrapper for aten::squeeze
    Returns a tensor with all specified dimensions of input of size 1 removed.

    If dim is not specified, all dimensions of size 1 are removed.
    If dim is specified, only that dimension is squeezed (if it has size 1).
    """
    logger.debug("GEMS SQUEEZE")

    # Compute the new shape after squeezing
    if dim is None:
        # Remove all dimensions of size 1
        new_shape = tuple(d for d in inp.shape if d != 1)
    else:
        # Handle negative dim
        dim = dim if dim >= 0 else inp.dim() + dim
        # Remove only the specified dimension if it has size 1
        new_shape = list(inp.shape)
        if inp.shape[dim] == 1:
            new_shape.pop(dim)
        new_shape = tuple(new_shape)

    # If there's no change in shape, just return the input
    if new_shape == inp.shape:
        return inp

    # If input has no elements, just return empty tensor with new shape
    if inp.numel() == 0:
        return inp.reshape(new_shape)

    # Try to use view first (most efficient - no data copy)
    try:
        return inp.view(new_shape)
    except RuntimeError:
        # If view fails (e.g., non-contiguous tensor), fall back to Triton copy
        pass

    # Fallback: use Triton to copy data with new shape
    out = torch.empty(new_shape, dtype=inp.dtype, device=inp.device)
    if out.numel() == 0:
        return out
    overload = _squeeze_identity.instantiate(inp.dim())
    overload(inp, out0=out)
    return out


def squeeze_(inp, dim=None):
    """
    In-place version of squeeze.
    Note: PyTorch's squeeze_ is deprecated, but we implement it for completeness.
    """
    logger.debug("GEMS SQUEEZE_")
    # For in-place, we need to copy back to the original tensor
    result = squeeze(inp, dim=dim)
    inp.copy_(result)
    return inp