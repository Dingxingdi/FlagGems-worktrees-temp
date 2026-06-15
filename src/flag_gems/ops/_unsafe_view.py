import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.autotune(
    key=["n_elements"],
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=4, num_warps=1),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_stages=4, num_warps=8),
    ],
)
@triton.jit
def _unsafe_view_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


def _unsafe_view(A: torch.Tensor, size: torch.Size) -> torch.Tensor:
    logger.debug("GEMS UNSAFE_VIEW")
    """
    Wrapper for aten::_unsafe_view
    Returns a tensor with the specified shape but shares memory with the original tensor.
    This implementation creates a new tensor with the target shape and copies the data.
    """
    # Handle empty tensors
    if A.numel() == 0:
        return torch.empty(size, dtype=A.dtype, device=A.device)

    # Calculate total elements in target shape
    target_numel = 1
    for s in size:
        target_numel *= s

    if target_numel != A.numel():
        raise RuntimeError(
            f"shape '{list(size)}' is invalid for input of size {A.numel()}"
        )

    out = torch.empty(size, dtype=A.dtype, device=A.device)
    n_elements = A.numel()

    # Ensure contiguous memory for efficient linear copy
    src = A.contiguous() if not A.is_contiguous() else A
    if not out.is_contiguous():
        out = out.contiguous()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device):
        _unsafe_view_kernel[grid](src, out, n_elements)
    return out