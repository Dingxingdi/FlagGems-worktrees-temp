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


def transpose_copy(A: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    logger.debug("GEMS TRANSPOSE_COPY")

    # Validate dimensions
    dim0 = dim0 if dim0 >= 0 else A.dim() + dim0
    dim1 = dim1 if dim1 >= 0 else A.dim() + dim1

    assert dim0 >= 0 and dim0 < A.dim(), f"dim0 {dim0} out of range [0, {A.dim()})"
    assert dim1 >= 0 and dim1 < A.dim(), f"dim1 {dim1} out of range [0, {A.dim()})"

    # Handle trivial case: same dimension or size 1
    if dim0 == dim1 or A.size(dim0) == 1 or A.size(dim1) == 1:
        return A.clone()

    # Create output tensor
    out_shape = list(A.shape)
    out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
    out = torch.empty(out_shape, dtype=A.dtype, device=A.device)

    # Create transposed strides for the input (view)
    strides = list(A.stride())
    strides[dim0], strides[dim1] = strides[dim1], strides[dim0]

    # Create a StridedBuffer that represents the transposed view
    transposed_A = StridedBuffer(A, shape=tuple(out_shape), strides=tuple(strides))

    # Copy data using pointwise_dynamic
    overload = copy_func.instantiate(A.ndim)
    overload(transposed_A, out0=out)

    return out