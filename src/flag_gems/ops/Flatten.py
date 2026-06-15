import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _compute_flatten_shape(shape, start_dim, end_dim):
    """Compute the output shape for flatten operation."""
    ndim = len(shape)
    if end_dim < 0:
        end_dim = ndim + end_dim
    if start_dim < 0:
        start_dim = ndim + start_dim

    flattened_dim = 1
    for i in range(start_dim, end_dim + 1):
        flattened_dim *= shape[i]

    output_shape = list(shape[:start_dim]) + [flattened_dim]
    if end_dim < ndim - 1:
        output_shape.extend(shape[end_dim + 1:])

    return tuple(output_shape)


# Simple identity kernel - used to verify Triton is properly imported
@triton.jit
def _identity_kernel(x):
    return x


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def _flatten_kernel(
    output_ptr,
    input_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to flatten a tensor by copying elements to a new shape.
    Uses a simple 1D index mapping for the copy operation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    local_offsets = tl.arange(0, BLOCK_SIZE)
    idxs = block_start + local_offsets
    mask = idxs < numel

    # Load and store using vectorized operations
    vals = tl.load(input_ptr + local_offsets, mask=mask)
    tl.store(output_ptr + local_offsets, vals, mask=mask)


def flatten(input: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:
    """
    Flattens a tensor by reshaping it into a one-dimensional tensor.
    If start_dim or end_dim are passed, only dimensions starting with
    start_dim and ending with end_dim are flattened.
    """
    logger.debug("GEMS FLATTEN")

    ndim = input.ndim
    if ndim == 0:
        return input

    # Normalize dimensions
    if end_dim < 0:
        end_dim = ndim + end_dim
    if start_dim < 0:
        start_dim = ndim + start_dim

    start_dim = max(0, min(start_dim, ndim - 1))
    end_dim = max(start_dim, min(end_dim, ndim - 1))

    # Compute output shape
    output_shape = _compute_flatten_shape(input.shape, start_dim, end_dim)

    # For contiguous tensors, try reshape first (most efficient)
    if input.is_contiguous():
        try:
            return input.reshape(output_shape)
        except RuntimeError:
            pass

    # For non-contiguous case, use Triton kernel for the copy
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    numel = output.numel()

    if numel > 0:
        # Flatten input to 1D for the kernel, then reshape output
        # This works because we're doing a simple copy
        input_1d = input.reshape(-1)

        # Use the kernel
        grid = (triton.cdiv(numel, 512),)
        _flatten_kernel[grid](
            output,
            input_1d,
            numel,
        )

    return output


def flatten_(input: torch.Tensor, start_dim: int = 0, end_dim: int = -1) -> torch.Tensor:
    """
    In-place flatten. Since flatten changes the shape, this creates a flattened copy
    and copies it back to the input tensor.
    """
    logger.debug("GEMS FLATTEN_")

    result = flatten(input, start_dim, end_dim)
    input.resize_(result.shape)
    input.copy_(result)
    return input