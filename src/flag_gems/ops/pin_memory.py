import logging

import torch

logger = logging.getLogger(__name__)


def pin_memory(inp, device=None):
    """Pin the memory of the input tensor for faster CPU-GPU data transfer.

    This is primarily a CPU-side operation that pins the memory to make it
    page-locked for faster async data transfers to GPU.
    """
    logger.debug("GEMS PIN_MEMORY")
    # Delegate to PyTorch's implementation via redispatch to avoid recursion
    return torch.ops.aten.pin_memory.default(inp, device)