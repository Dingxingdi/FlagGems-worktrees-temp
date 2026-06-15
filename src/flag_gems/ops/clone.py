import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _clone_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def clone(inp: torch.Tensor, memory_format: Optional[torch.memory_format] = None):
    """
    Returns a copy of the input tensor.

    Args:
        inp: The input tensor to clone.
        memory_format: The desired memory format of the returned tensor.
            Default is torch.preserve_format.
    """
    if memory_format is None:
        memory_format = torch.preserve_format

    logger.debug("GEMS CLONE")

    n_elements = inp.numel()
    if n_elements == 0:
        # Handle empty tensors
        if memory_format == torch.contiguous_format:
            return torch.empty_like(inp, memory_format=torch.contiguous_format)
        else:
            return torch.empty_strided(
                inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
            )

    # Handle memory_format
    if memory_format == torch.preserve_format:
        # Preserve the original memory format
        if inp.is_contiguous(memory_format=torch.preserve_format):
            # Use strided output to preserve the memory layout
            out = torch.empty_strided(
                inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
            )
            # Flatten the input for the kernel (contiguous view for loading)
            src = inp.flatten()
            dst = out.flatten()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            with torch_device_fn.device(inp.device):
                _clone_kernel[grid](src, dst, n_elements, BLOCK_SIZE=1024)
            return out
        else:
            # If not contiguous in preserve format, fall back to PyTorch
            return torch.clone(inp, memory_format=memory_format)
    elif memory_format == torch.contiguous_format:
        # Make the result contiguous
        out = torch.empty_like(inp, memory_format=torch.contiguous_format)
        src = inp.flatten()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(inp.device):
            _clone_kernel[grid](src, out, n_elements, BLOCK_SIZE=1024)
        return out
    else:
        # For other memory formats, fall back to PyTorch
        return torch.clone(inp, memory_format=memory_format)