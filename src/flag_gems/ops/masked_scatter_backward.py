import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    key=["numel"],
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    ],
)
@triton.jit
def masked_scatter_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    output_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load the indices in the grad_output where we should gather from
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

    # Load values from grad_output at those indices
    # grad_output is flattened, so we use simple linear indexing
    values = tl.load(grad_output_ptr + indices, mask=mask, other=0.0)

    # Store to output
    tl.store(output_ptr + offsets, values, mask=mask)


def masked_scatter_backward(grad_output, mask, sizes):
    logger.debug("GEMS MASKED_SCATTER_BACKWARD")

    # sizes should be a list/tuple specifying the output shape (typically 1D)
    # The output contains grad_output[mask] in row-major order

    # Convert sizes to a list if it's a tuple
    if isinstance(sizes, (tuple, list)):
        sizes = list(sizes)
    else:
        sizes = [sizes]

    # Number of True elements in mask
    mask_numel = mask.sum().item()

    # Expected output size from sizes
    output_numel = 1
    for s in sizes:
        output_numel *= s

    # Validate
    assert output_numel == mask_numel, \
        f"Output size {output_numel} does not match number of True elements in mask {mask_numel}"

    if output_numel == 0:
        return torch.empty(sizes, dtype=grad_output.dtype, device=grad_output.device)

    # Get the indices where mask is True in row-major order (flattened)
    # These are the positions in the flattened grad_output that we need to gather
    mask_flat = mask.flatten()
    indices = torch.nonzero(mask_flat, as_tuple=False).flatten().to(grad_output.device)

    # Flatten grad_output for linear indexing
    grad_output_flat = grad_output.flatten()

    # Allocate output
    output = torch.empty(sizes, dtype=grad_output.dtype, device=grad_output.device)

    # Launch Triton kernel - let autotune handle the BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(output_numel, meta["BLOCK_SIZE"]),)

    masked_scatter_backward_kernel[grid](
        grad_output_flat,
        indices,
        output,
        output_numel,
    )

    return output


def masked_scatter_backward_(grad_output, mask, sizes):
    """In-place version is not supported for this backward function"""
    return masked_scatter_backward(grad_output, mask, sizes)