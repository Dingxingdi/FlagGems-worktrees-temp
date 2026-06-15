import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def squeeze_kernel(input_ptr, output_ptr, numel, BLOCK_SIZE: tl.constexpr):
    """Squeeze kernel that copies data element by element."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load from input
    input_ptrs = input_ptr + offsets
    values = tl.load(input_ptrs, mask=mask, other=0.0)

    # Store to output
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, values, mask=mask)


def squeeze(inp: torch.Tensor, dim=None) -> torch.Tensor:
    """Squeeze operation that removes dimensions of size 1.

    Args:
        inp: Input tensor
        dim: Optional dimension(s) to squeeze. If None, removes all dims of size 1.

    Returns:
        Tensor with specified dimensions of size 1 removed.
    """
    logger.debug("GEMS SQUEEZE")

    numel = inp.numel()

    # For empty tensors or scalars, just return as is
    if numel == 0 or inp.dim() == 0:
        return inp.clone()

    # Compute output shape
    if dim is None:
        # Remove all dimensions of size 1
        output_shape = tuple(d for d in inp.shape if d != 1)
        # Note: if all dimensions are 1, output_shape becomes ()
        # which is what PyTorch returns for squeeze of (1,) -> scalar
    else:
        # Handle single dimension case
        if isinstance(dim, int):
            dims_to_squeeze = (dim,)
        else:
            dims_to_squeeze = tuple(dim)

        # Normalize negative dimensions
        ndims = inp.dim()
        normalized_dims = []
        for d in dims_to_squeeze:
            if d < 0:
                d = d + ndims
            normalized_dims.append(d)

        output_shape = []
        for i, size in enumerate(inp.shape):
            if i in normalized_dims:
                # Only squeeze if the dimension has size 1
                if size != 1:
                    output_shape.append(size)
                # If dim is specified but size != 1, keep the dimension
            else:
                output_shape.append(size)
        output_shape = tuple(output_shape)

    # Handle case where output shape is same as input
    if output_shape == inp.shape:
        return inp.clone()

    # For contiguous tensors, reshape gives us a view (no data copy)
    # which is the same behavior as PyTorch's squeeze
    if inp.is_contiguous():
        return inp.reshape(output_shape)

    # For non-contiguous tensors, we need to copy data using Triton kernel
    inp_flat = inp.reshape(-1)
    out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    # Run Triton kernel to copy data
    BLOCK_SIZE = 128
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    squeeze_kernel[grid](inp_flat, out, numel, BLOCK_SIZE)

    return out


def squeeze_(inp: torch.Tensor, dim=None) -> torch.Tensor:
    """In-place squeeze operation.

    Note: This creates a new tensor with squeezed shape rather than
    modifying in-place due to Triton limitations.
    """
    logger.debug("GEMS SQUEEZE_")
    return squeeze(inp, dim)