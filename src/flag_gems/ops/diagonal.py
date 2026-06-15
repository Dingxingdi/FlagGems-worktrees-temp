import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


@triton.jit
def diagonal_kernel(
    data_ptr,
    output_ptr,
    # Number of diagonal elements per batch
    diag_len,
    # Total number of batch entries
    num_batches,
    # Size of dim1
    dim1_size,
    # Size of dim2
    dim2_size,
    # Stride for dim1
    stride_dim1,
    # Stride for dim2
    stride_dim2,
    # Total input size in elements
    input_size,
    # Number of batch dimensions
    num_batch_dims: tl.constexpr,
    # Batch stride 0
    batch_stride0,
    # Batch stride 1
    batch_stride1,
    # Batch dim 0 size
    batch_dim0_size,
    offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one diagonal element for one batch entry
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < input_size

    # Calculate batch_id and diag_id from linear index
    batch_id = idx // diag_len
    diag_id = idx % diag_len

    batch_mask = batch_id < num_batches

    # Compute row and col indices based on offset
    if offset >= 0:
        row_idx = diag_id
        col_idx = diag_id + offset
    else:
        row_idx = diag_id - offset
        col_idx = diag_id

    # Check if indices are valid
    valid = (row_idx < dim1_size) & (col_idx < dim2_size)
    mask = mask & batch_mask & valid

    # Compute linear offset into input tensor
    if num_batch_dims == 1:
        input_offset = batch_id * batch_stride0 + row_idx * stride_dim1 + col_idx * stride_dim2
    elif num_batch_dims >= 2:
        # For 2+ batch dims: compute batch0 and batch1 from batch_id
        batch0 = batch_id % batch_dim0_size
        batch1 = batch_id // batch_dim0_size
        input_offset = batch0 * batch_stride0 + batch1 * batch_stride1 + row_idx * stride_dim1 + col_idx * stride_dim2
    else:
        # No batch dims
        input_offset = row_idx * stride_dim1 + col_idx * stride_dim2

    values = tl.load(data_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + idx, values, mask=mask)


def diagonal(input, offset=0, dim1=-2, dim2=-1):
    """
    Extracts the diagonal from a tensor.

    This is a Triton implementation for torch.diagonal and torch.linalg.diagonal.
    """
    logger.debug("GEMS DIAGONAL")

    # Normalize negative dimensions
    dim1 = dim1 if dim1 >= 0 else input.ndim + dim1
    dim2 = dim2 if dim2 >= 0 else input.ndim + dim2

    # Get the size of the two dimensions
    dim1_size = input.shape[dim1]
    dim2_size = input.shape[dim2]

    # Calculate the diagonal length
    if offset >= 0:
        diag_len = min(dim1_size, dim2_size - offset)
    else:
        diag_len = min(dim1_size + offset, dim2_size)

    if diag_len <= 0:
        # Return empty tensor
        return torch.empty(0, dtype=input.dtype, device=input.device)

    # Calculate output shape
    # The output removes dim2 and replaces dim1 with the diagonal dimension
    output_shape = list(input.shape)
    output_shape.pop(dim2)
    output_shape[dim1] = diag_len
    output_shape = tuple(output_shape)

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Get strides
    strides = input.stride()
    stride_dim1 = strides[dim1]
    stride_dim2 = strides[dim2]

    # Calculate total number of batch entries and batch strides
    num_batch_dims = dim1
    if num_batch_dims >= 1:
        batch_stride0 = strides[0]
        num_batches = input.shape[0]
        batch_dim0_size = input.shape[0]
        if num_batch_dims >= 2:
            batch_stride1 = strides[1]
            num_batches = num_batches * input.shape[1]
        else:
            batch_stride1 = 0
    else:
        batch_stride0 = 0
        batch_stride1 = 0
        num_batches = 1
        batch_dim0_size = 1

    # Total elements to process
    total_elements = num_batches * diag_len

    BLOCK_SIZE = 128
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        diagonal_kernel[grid](
            input,
            output,
            diag_len,
            num_batches,
            dim1_size,
            dim2_size,
            stride_dim1,
            stride_dim2,
            total_elements,
            num_batch_dims,
            batch_stride0,
            batch_stride1,
            batch_dim0_size,
            offset,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logger.debug("GEMS diagonal backward")
    grad_input = torch.zeros(
        input_sizes, dtype=grad_output.dtype, device=grad_output.device
    )
    diag = torch.diagonal(grad_input, offset, dim1, dim2)
    copy_func.instantiate(grad_output.ndim)(grad_output, out0=diag)
    return grad_input
