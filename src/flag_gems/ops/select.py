import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def select_kernel(
    inp_ptr,
    out_ptr,
    total_elements,
    dim_size,
    dim_prod_post,
    index,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_elements
    idx = block_start + offsets

    # Calculate indices for the output tensor (which has one less dimension)
    # out_idx = pre_idx * dim_prod_post + post_idx
    # We need to compute the corresponding input index
    # inp_idx = pre_idx * dim_size * dim_prod_post + index * dim_prod_post + post_idx

    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post
    inp_idx = pre_idx * dim_size * dim_prod_post + index * dim_prod_post + post_idx

    # Load from input and store to output
    inp_data = tl.load(inp_ptr + inp_idx, mask=mask)
    tl.store(out_ptr + idx, inp_data, mask=mask)


def select(inp, dim, index):
    logger.debug("GEMS SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)

    # Compute output shape (remove the selected dimension)
    out_shape = list(inp.shape)
    del out_shape[dim]

    if inp.numel() == 0:
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Handle edge case: 0-sized dimension after removal
    if 0 in out_shape:
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    inp = inp.contiguous()

    # Total elements in output (input elements divided by selected dimension size)
    out_numel = inp.numel() // inp.size(dim)
    dim_size = inp.size(dim)

    dim_prod_post = 1
    for d in range(dim + 1, inp.ndim):
        dim_prod_post *= inp.size(d)

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(out_numel, BLOCK_SIZE),)

    select_kernel[grid](
        inp,
        out,
        out_numel,
        dim_size,
        dim_prod_post,
        index,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out