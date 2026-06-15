import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def fft_ifftshift_kernel(
    input_ptr,
    output_ptr,
    total_elements,
    ndim,
    shift_dim_0,
    shift_dim_1,
    shift_dim_2,
    shift_dim_3,
    shift_dim_4,
    dim0_size,
    dim1_size,
    dim2_size,
    dim3_size,
    dim4_size,
    dim0_stride,
    dim1_stride,
    dim2_stride,
    dim3_stride,
    dim4_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute multi-dimensional indices from linear offset
    # For row-major (C) order: last dimension changes fastest
    idx = offsets

    # Dimension 4 (last in row-major)
    dim4_idx = idx % dim4_size
    idx = idx // dim4_size

    if shift_dim_4 > 0:
        dim4_idx = (dim4_idx + shift_dim_4) % dim4_size

    # Dimension 3
    dim3_idx = idx % dim3_size
    idx = idx // dim3_size

    if shift_dim_3 > 0:
        dim3_idx = (dim3_idx + shift_dim_3) % dim3_size

    # Dimension 2
    dim2_idx = idx % dim2_size
    idx = idx // dim2_size

    if shift_dim_2 > 0:
        dim2_idx = (dim2_idx + shift_dim_2) % dim2_size

    # Dimension 1
    dim1_idx = idx % dim1_size
    idx = idx // dim1_size

    if shift_dim_1 > 0:
        dim1_idx = (dim1_idx + shift_dim_1) % dim1_size

    # Dimension 0 (first in row-major)
    dim0_idx = idx % dim0_size

    if shift_dim_0 > 0:
        dim0_idx = (dim0_idx + shift_dim_0) % dim0_size

    # Compute linear source index
    src_offset = (
        dim0_idx * dim0_stride
        + dim1_idx * dim1_stride
        + dim2_idx * dim2_stride
        + dim3_idx * dim3_stride
        + dim4_idx * dim4_stride
    )

    # Load from input and store to output
    val = tl.load(input_ptr + src_offset, mask=mask)
    tl.store(output_ptr + offsets, val, mask=mask)


def fft_ifftshift(input: torch.Tensor, dim=None) -> torch.Tensor:
    logger.debug("GEMS FFT_IFFTSHIFT")

    # Handle default case: shift all dimensions
    if dim is None:
        shift_dims = list(range(input.ndim))
    elif isinstance(dim, int):
        shift_dims = [dim]
    else:
        shift_dims = list(dim)

    # Handle negative dimensions
    shift_dims = [d if d >= 0 else d + input.ndim for d in shift_dims]

    # Compute shift amounts: dim_size // 2 for each dimension
    shift_dict = {d: input.shape[d] // 2 for d in shift_dims}

    output = torch.empty_like(input)
    total_elements = input.numel()

    if total_elements == 0:
        return output

    # Define block size
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    # Get shape and stride - pad to 5 dimensions
    ndim = input.ndim
    shape = list(input.shape) + [1] * (5 - ndim)
    stride = list(input.stride()) + [0] * (5 - ndim)

    fft_ifftshift_kernel[grid](
        input,
        output,
        total_elements,
        ndim,
        shift_dict.get(0, 0),
        shift_dict.get(1, 0),
        shift_dict.get(2, 0),
        shift_dict.get(3, 0),
        shift_dict.get(4, 0),
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        stride[0],
        stride[1],
        stride[2],
        stride[3],
        stride[4],
        BLOCK_SIZE,
    )

    return output