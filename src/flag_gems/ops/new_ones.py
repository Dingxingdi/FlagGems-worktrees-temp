import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.ones import ones_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def new_ones(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None
):
    """Returns a tensor filled with ones with the same dtype and device as self.

    Args:
        self: The tensor to get dtype and device from
        size: Shape of the output tensor
        dtype: The desired dtype of the output tensor. If None, uses self.dtype
        layout: The desired layout of the output tensor
        device: The desired device of the output tensor. If None, uses self.device
        pin_memory: Whether to use pinned memory

    Returns:
        A tensor filled with ones
    """
    logger.debug("GEMS NEW_ONES")
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype

    out = torch.empty(size, device=device, dtype=dtype)
    N = out.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(device):
        ones_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out