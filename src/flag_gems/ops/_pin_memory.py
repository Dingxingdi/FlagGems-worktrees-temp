import logging

import torch

from flag_gems.ops.copy import copy_

logger = logging.getLogger(__name__)


def _pin_memory(inp, device=None):
    """Pin memory operator - returns a copy of the input tensor."""
    logger.debug("GEMS PIN_MEMORY")
    # Create output tensor with same shape, dtype, and device
    out = torch.empty_like(inp)
    copy_(out, inp)
    return out


def _pin_memory_(inp, device=None):
    """In-place pin memory operator (alias to copy_)."""
    logger.debug("GEMS PIN_MEMORY_")
    copy_(inp, inp)
    return inp