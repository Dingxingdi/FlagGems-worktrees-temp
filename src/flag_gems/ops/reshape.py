import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# Define a Triton kernel for copying data during reshape
# This is used when the input is not contiguous
@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _reshape_copy_kernel(src):
    """Triton kernel for copying data during reshape."""
    return src


def reshape(inp, shape):
    """
    Reshape tensor to given shape.

    If the input is already contiguous and the reshape is compatible,
    this returns a view. Otherwise, it creates a new tensor and copies
    the data using Triton kernel.
    """
    logger.debug("GEMS RESHAPE")

    # Handle -1 in shape (inferred dimension)
    if -1 in shape:
        # Calculate the actual size for -1
        total_elements = inp.numel()
        known_elements = 1
        minus_one_count = 0
        for s in shape:
            if s == -1:
                minus_one_count += 1
            else:
                known_elements *= s

        if minus_one_count == 1 and total_elements % known_elements == 0:
            shape = tuple(s if s != -1 else total_elements // known_elements for s in shape)
        else:
            raise ValueError(f"Cannot infer shape {shape} for tensor with {total_elements} elements")

    # If the shape is the same, just return input
    if tuple(shape) == tuple(inp.shape):
        return inp

    # Try to use view if possible (when tensor is contiguous)
    if inp.is_contiguous():
        try:
            return inp.view(shape)
        except RuntimeError:
            pass

    # Need to copy data to new shape
    # First make contiguous using our Triton kernel
    inp_contig = torch.empty_like(inp)
    _reshape_copy_kernel(inp, out0=inp_contig)

    # Now we can view it to the target shape
    return inp_contig.view(shape)