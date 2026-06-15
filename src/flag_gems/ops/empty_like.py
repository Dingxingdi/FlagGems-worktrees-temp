import logging

import torch

logger = logging.getLogger(__name__)


def empty_like(
    x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    logger.debug("GEMS EMPTY_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    # Use torch.empty + reshape to avoid recursion with GEMS intercepting torch.empty_like
    size = x.size()
    out = torch.empty(size, device=device, dtype=dtype)
    return out