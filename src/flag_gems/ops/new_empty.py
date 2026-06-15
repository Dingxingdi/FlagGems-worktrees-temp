import logging

import torch

logger = logging.getLogger(__name__)


def new_empty(x, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    """Create a tensor filled with uninitialized data.

    Args:
        x: the tensor to use for determining dtype and device
        size: a list, tuple, or torch.Size of integers defining the shape
        dtype: the desired type of returned tensor. Default: if None, same as x.dtype
        layout: the desired layout of returned Tensor. Default: torch.strided
        device: the desired device of returned tensor. Default: if None, same as x.device
        pin_memory: if set, returned tensor would be allocated in the pinned
            memory. Works only for CPU tensors. Default: False

    Returns:
        A tensor with uninitialized data
    """
    logger.debug("GEMS NEW_EMPTY")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype

    return torch.empty(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)