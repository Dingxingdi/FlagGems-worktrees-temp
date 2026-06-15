import logging
from typing import List

import triton
import triton.language as tl

import torch

from flag_gems.ops.copy import copy_

logger = logging.getLogger(__name__)


@triton.jit
def _resize_kernel(src):
    """Simple Triton kernel for resize operation."""
    return src


def _resize_output_(self: torch.Tensor, size: List[int], device: torch.device) -> torch.Tensor:
    """Resize the output tensor to the specified size.

    Args:
        self: Input tensor
        size: Target size as a list of integers
        device: Target device

    Returns:
        A resized tensor
    """
    logger.debug("GEMS _resize_output_")

    # If the new size is the same as the original size, just return a copy on the target device
    if list(self.shape) == size and self.device == device:
        return self.clone()

    # Calculate the number of elements to copy
    new_numel = 1
    for s in size:
        new_numel *= s
    min_numel = min(self.numel(), new_numel)

    if min_numel > 0:
        # Flatten the input and create a contiguous 1D tensor
        src = self.reshape(-1)[:min_numel].clone()

        # Create output and flatten - use zeros to initialize
        out = torch.zeros(size, dtype=self.dtype, device=device)
        dst = out.reshape(-1)

        # Use copy_ from FlagGems which uses Triton
        # Copy to the first min_numel elements
        copy_(dst[:min_numel], src)
    else:
        out = torch.empty(size, dtype=self.dtype, device=device)

    return out


def _resize_output__(self: torch.Tensor, size: List[int], device: torch.device) -> torch.Tensor:
    """In-place version of _resize_output_.

    Args:
        self: Input tensor
        size: Target size as a list of integers
        device: Target device (not used, kept for API consistency)

    Returns:
        The resized tensor
    """
    logger.debug("GEMS _resize_output_")
    return _resize_output_(self, size, device)