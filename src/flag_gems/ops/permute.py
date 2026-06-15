import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def permute_kernel(
    input_ptr,
    output_ptr,
    # Output shape (max 5D)
    shape_0,
    shape_1,
    shape_2,
    shape_3,
    shape_4,
    # Output strides
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    out_stride_4,
    # Input strides
    in_stride_0,
    in_stride_1,
    in_stride_2,
    in_stride_3,
    in_stride_4,
    # Dimension mapping (permute dims)
    dim0,
    dim1,
    dim2,
    dim3,
    dim4,
    # Number of elements
    numel: tl.constexpr,
    rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the position within the output tensor
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Create the range of indices for this block
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Calculate output multi-indices from linear offset
    # Then map to input indices based on permute dims
    if rank == 1:
        i0 = offs
        # output[i0] = input[dims[0]]
        src_i0 = i0 * dim0
        src_offset = src_i0
    elif rank == 2:
        i0 = offs // shape_1
        i1 = offs % shape_1
        # output[i0, i1] = input[dims[0], dims[1]]
        src_i0 = i0 * dim0
        src_i1 = i1 * dim1
        src_offset = src_i0 + src_i1
    elif rank == 3:
        s1 = shape_1
        s2 = shape_2
        i0 = offs // (s1 * s2)
        i1 = (offs // s2) % s1
        i2 = offs % s2
        src_i0 = i0 * dim0
        src_i1 = i1 * dim1
        src_i2 = i2 * dim2
        src_offset = src_i0 + src_i1 + src_i2
    elif rank == 4:
        s1 = shape_1
        s2 = shape_2
        s3 = shape_3
        i0 = offs // (s1 * s2 * s3)
        i1 = (offs // (s2 * s3)) % s1
        i2 = (offs // s3) % s2
        i3 = offs % s3
        src_i0 = i0 * dim0
        src_i1 = i1 * dim1
        src_i2 = i2 * dim2
        src_i3 = i3 * dim3
        src_offset = src_i0 + src_i1 + src_i2 + src_i3
    else:  # rank == 5
        s1 = shape_1
        s2 = shape_2
        s3 = shape_3
        s4 = shape_4
        i0 = offs // (s1 * s2 * s3 * s4)
        i1 = (offs // (s2 * s3 * s4)) % s1
        i2 = (offs // (s3 * s4)) % s2
        i3 = (offs // s4) % s3
        i4 = offs % s4
        src_i0 = i0 * dim0
        src_i1 = i1 * dim1
        src_i2 = i2 * dim2
        src_i3 = i3 * dim3
        src_i4 = i4 * dim4
        src_offset = src_i0 + src_i1 + src_i2 + src_i3 + src_i4

    # Load from input and store to output
    vals = tl.load(input_ptr + src_offset, mask=mask)
    out_offset = offs
    tl.store(output_ptr + out_offset, vals, mask=mask)


def permute(input: torch.Tensor, dims) -> torch.Tensor:
    """
    Permute the dimensions of a tensor.

    Args:
        input: Input tensor
        dims: Permutation of dimensions

    Returns:
        Permuted tensor
    """
    logger.debug("GEMS PERMUTE")

    # Validate dims
    dims = list(dims)
    ndim = input.ndim

    assert len(dims) == ndim, (
        f"Number of dimensions in dims ({len(dims)}) must match "
        f"input.ndim ({ndim})"
    )

    # Normalize dims to positive
    dims = [d if d >= 0 else d + ndim for d in dims]

    # Check that dims is a valid permutation
    assert sorted(dims) == list(range(ndim)), (
        f"dims must be a valid permutation of input dimensions, got {dims}"
    )

    # Compute output shape
    output_shape = tuple(input.shape[d] for d in dims)

    # Get input strides
    input_strides = input.stride()
    input_shape = input.shape

    # Compute the stride for each output dimension in the input tensor
    # For output dimension i, it corresponds to input dimension dims[i]
    # So the stride in the input is input.stride()[dims[i]]
    input_stride_list = list(input_strides)

    # Create output tensor
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Get output strides (these are for the linear storage)
    output_strides = output.stride()

    # For the kernel, we need the mapping:
    # output[i0, i1, ...] = input[dims[0], dims[1], ...]
    # The stride in input for output dimension i is input_stride[dims[i]]
    dim_strides = [input_stride_list[d] for d in dims]

    # Handle different ranks
    numel = input.numel()

    if numel == 0:
        return output

    # Set up kernel parameters based on rank
    BLOCK_SIZE = 512
    num_warps = 8
    num_ctas = 1

    if ndim == 1:
        shape0 = output_shape[0] if len(output_shape) > 0 else 1
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        permute_kernel[grid](
            input,
            output,
            shape0,
            0,
            0,
            0,
            0,  # shape 1-5
            output_strides[0] if len(output_strides) > 0 else 0,
            0,
            0,
            0,
            0,  # output strides 1-5
            input_strides[0] if len(input_strides) > 0 else 0,
            0,
            0,
            0,
            0,  # input strides 1-5
            dim_strides[0] if len(dim_strides) > 0 else 0,
            0,
            0,
            0,
            0,  # dim strides 1-5
            numel,
            1,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
    elif ndim == 2:
        shape0, shape1 = output_shape
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        permute_kernel[grid](
            input,
            output,
            shape0,
            shape1,
            0,
            0,
            0,
            output_strides[0],
            output_strides[1] if len(output_strides) > 1 else 0,
            0,
            0,
            0,
            input_strides[0],
            input_strides[1] if len(input_strides) > 1 else 0,
            0,
            0,
            0,
            dim_strides[0],
            dim_strides[1] if len(dim_strides) > 1 else 0,
            0,
            0,
            0,
            numel,
            2,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
    elif ndim == 3:
        shape0, shape1, shape2 = output_shape
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        permute_kernel[grid](
            input,
            output,
            shape0,
            shape1,
            shape2,
            0,
            0,
            output_strides[0],
            output_strides[1],
            output_strides[2] if len(output_strides) > 2 else 0,
            0,
            0,
            input_strides[0],
            input_strides[1],
            input_strides[2] if len(input_strides) > 2 else 0,
            0,
            0,
            dim_strides[0],
            dim_strides[1],
            dim_strides[2] if len(dim_strides) > 2 else 0,
            0,
            0,
            numel,
            3,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
    elif ndim == 4:
        shape0, shape1, shape2, shape3 = output_shape
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        permute_kernel[grid](
            input,
            output,
            shape0,
            shape1,
            shape2,
            shape3,
            0,
            output_strides[0],
            output_strides[1],
            output_strides[2],
            output_strides[3] if len(output_strides) > 3 else 0,
            0,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            input_strides[3] if len(input_strides) > 3 else 0,
            0,
            dim_strides[0],
            dim_strides[1],
            dim_strides[2],
            dim_strides[3] if len(dim_strides) > 3 else 0,
            0,
            numel,
            4,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
    elif ndim == 5:
        shape0, shape1, shape2, shape3, shape4 = output_shape
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        permute_kernel[grid](
            input,
            output,
            shape0,
            shape1,
            shape2,
            shape3,
            shape4,
            output_strides[0],
            output_strides[1],
            output_strides[2],
            output_strides[3],
            output_strides[4] if len(output_strides) > 4 else 0,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            input_strides[3],
            input_strides[4] if len(input_strides) > 4 else 0,
            dim_strides[0],
            dim_strides[1],
            dim_strides[2],
            dim_strides[3],
            dim_strides[4] if len(dim_strides) > 4 else 0,
            numel,
            5,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        # For other ranks, fall back to PyTorch
        output = input.permute(dims).contiguous()

    return output