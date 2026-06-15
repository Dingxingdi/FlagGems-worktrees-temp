import logging

import torch

logger = logging.getLogger(__name__)


def view(inp: torch.Tensor, *args):
    """
    Wrapper for aten::view
    Returns a new tensor with the same data but different shape.
    """
    logger.debug("GEMS VIEW")
    # Handle the case where shape is passed as a single argument (list/tuple)
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        shape = list(args[0])
    else:
        shape = list(args)

    # Calculate total number of elements
    numel = 1
    for s in shape:
        numel *= s

    if numel != inp.numel():
        raise RuntimeError(
            f"view size is not compatible with input tensor's size and stride. "
            f"Input has {inp.numel()} elements, view shape {shape} has {numel} elements"
        )

    # For contiguous tensors, we can create a view using as_strided
    # with row-major (C) strides
    if inp.is_contiguous():
        # Calculate row-major strides
        strides = []
        stride = 1
        for s in reversed(shape):
            strides.insert(0, stride)
            stride *= s
        return inp.as_strided(shape, strides, inp.storage_offset())

    # For non-contiguous tensors, try to create a view that respects
    # the original stride pattern. This is more complex and may fail
    # for some cases. We use a fallback to reshape which may copy.
    return inp.reshape(shape)