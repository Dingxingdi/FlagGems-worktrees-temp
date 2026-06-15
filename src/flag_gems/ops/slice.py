import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def slice_kernel(
    output_ptr,
    input_ptr,
    total_elements,
    input_dim_size,
    dim_prod_post,
    start,
    step,
    output_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets
    pre_idx = idx // (output_dim_size * dim_prod_post)
    out_dim_idx = (idx // dim_prod_post) % output_dim_size
    post_idx = idx % dim_prod_post

    # Map output indices to input indices
    in_dim_idx = start + out_dim_idx * step
    input_idx = pre_idx * input_dim_size * dim_prod_post + in_dim_idx * dim_prod_post + post_idx

    data = tl.load(input_ptr + input_idx, mask=mask)
    tl.store(output_ptr + idx, data, mask=mask)


def slice(inp, dim=0, start=None, end=None, step=1):
    logger.debug("GEMS SLICE")
    assert step > 0, "slice step must be positive"

    # Normalize dim
    dim = dim % inp.ndim

    # Handle start and end
    dim_size = inp.size(dim)
    start = start if start is not None else 0
    end = end if end is not None else dim_size

    # Handle negative indices
    if start < 0:
        start = start % dim_size
    if end < 0:
        end = end % dim_size

    # Clamp start and end to valid range
    start = max(0, min(start, dim_size))
    end = max(0, min(end, dim_size))

    # Calculate output size
    slice_len = (end - start + step - 1) // step
    output_shape = list(inp.shape)
    output_shape[dim] = slice_len

    # Create output tensor
    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    # Handle empty slice case
    if slice_len <= 0:
        return output

    inp = inp.contiguous()

    total_elements = output.numel()

    # Calculate product of dimensions after the slice dim
    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    slice_kernel[grid](
        output,
        inp,
        total_elements,
        dim_size,
        dim_prod_post,
        start,
        step,
        slice_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output