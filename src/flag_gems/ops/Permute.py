import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def permute(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS PERMUTE")

    # Validate dims
    ndim = A.ndim
    if len(dims) != ndim:
        raise ValueError(
            f"permute(): invalid number of dimensions for tensor of dimension {ndim}: "
            f"expected {ndim} dims but got {len(dims)}"
        )

    # Normalize negative dims
    normalized_dims = []
    for dim in dims:
        if dim < 0:
            dim = dim + ndim
        if dim < 0 or dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], "
                f"but got {dim})"
            )
        normalized_dims.append(dim)

    # Check for duplicate dims
    if len(set(normalized_dims)) != ndim:
        raise ValueError("permute(): repeated dimension in dims")

    # Compute new shape and strides
    new_shape = tuple(A.shape[d] for d in normalized_dims)
    new_strides = tuple(A.stride()[d] for d in normalized_dims)

    # Handle trivial case: no change needed
    if tuple(normalized_dims) == tuple(range(ndim)):
        return A.clone()

    # Handle empty tensor
    if A.numel() == 0:
        return torch.empty(new_shape, dtype=A.dtype, device=A.device)

    # Create a strided view of the input
    permuted_A = StridedBuffer(A, shape=new_shape, strides=new_strides)

    # Allocate output
    out = torch.empty(new_shape, dtype=A.dtype, device=A.device)

    # Copy data using the generated kernel
    overload = copy_func.instantiate(A.ndim)
    overload(permuted_A, out0=out)

    return out


def permute_(A: torch.Tensor, dims) -> torch.Tensor:
    """In-place permute is not directly supported, but we can use out parameter"""
    logger.debug("GEMS PERMUTE_")

    result = permute(A, dims)
    A.copy_(result)
    return A