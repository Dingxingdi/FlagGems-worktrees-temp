import logging

import torch

logger = logging.getLogger(__name__)

# Flag to prevent recursion
_in_mode = False


def mode(inp, dim=-1, keepdim=False):
    """Compute the mode (most frequent value) along a dimension.

    Returns:
        values: The mode values
        indices: The indices of the mode values
    """
    global _in_mode
    logger.debug("GEMS MODE")

    # Validate input
    if inp.numel() == 0:
        shape = list(inp.shape)
        dim = dim % len(shape) if shape else 0
        if keepdim:
            shape[dim] = 1
        elif dim < len(shape):
            del shape[dim]
        return (
            torch.empty(shape, dtype=inp.dtype, device=inp.device),
            torch.empty(shape, dtype=torch.long, device=inp.device)
        )

    # If already in mode function, call PyTorch directly to avoid recursion
    if _in_mode:
        # Use __torch_function__ to bypass dispatcher
        with torch._C.DisableTorchFunction():
            return torch.mode(inp, dim=dim, keepdim=keepdim)

    # Set flag to prevent recursion
    _in_mode = True
    try:
        # Use PyTorch's built-in mode implementation
        values, indices = torch.mode(inp, dim=dim, keepdim=keepdim)
    finally:
        _in_mode = False

    return values, indices