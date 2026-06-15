import logging

import torch

logger = logging.getLogger(__name__)


def transpose_(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """
    In-place version of transpose. Swaps dimensions dim0 and dim1 of input tensor.

    Args:
        input: The input tensor (will be modified in-place)
        dim0: First dimension to swap
        dim1: Second dimension to swap

    Returns:
        The modified input tensor
    """
    logger.debug("GEMS TRANSPOSE_")

    # Normalize negative dimensions
    ndim = input.dim()
    dim0 = dim0 if dim0 >= 0 else dim0 + ndim
    dim1 = dim1 if dim1 >= 0 else dim1 + ndim

    # Validate dimensions
    if dim0 < 0 or dim0 >= ndim:
        raise RuntimeError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim0})"
        )
    if dim1 < 0 or dim1 >= ndim:
        raise RuntimeError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim1})"
        )

    # If swapping the same dimension, no change needed
    if dim0 == dim1:
        return input

    # Get current shape and stride
    shape = list(input.shape)
    stride = list(input.stride())

    # Swap shape and stride values
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]

    # Set the new shape and stride (in-place)
    input.as_strided_(shape, stride)

    return input