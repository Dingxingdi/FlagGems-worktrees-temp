import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def index_fill_kernel(
    inp,
    out,
    M,
    N,
    index_mask_ptr,
    value,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fill elements along the last dimension.

    Args:
        inp: input tensor (M, N) after dim_compress
        out: output tensor (M, N)
        M: number of elements in the "row" dimension (product of all dims except the indexed one)
        N: size along the indexed dimension
        index_mask_ptr: pointer to a boolean mask of size N, where True means fill with value
        value: the fill value
    """
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)

    row_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = row_offsets < M

    col_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N

    # Load the index mask for all columns in this block
    index_mask = tl.load(index_mask_ptr + col_offsets, mask=col_mask, other=0)

    # Compute offsets for loading/storing
    offsets = row_offsets * N + col_offsets
    mask = row_mask & col_mask

    # Load input values
    vals = tl.load(inp + offsets, mask=mask, other=0.0)

    # Where mask is True, use value; otherwise use input
    result = tl.where(index_mask, value, vals)

    tl.store(out + offsets, result, mask=mask)


def index_fill(inp, dim, index, value):
    logger.debug("GEMS INDEX FILL")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)

    dim = dim % inp.ndim

    # Create a boolean mask of size N (dim_size) with True at index positions
    N = inp.size(dim)
    index_mask = torch.zeros(N, dtype=torch.bool, device=inp.device)
    index_mask[index] = True

    # Use dim_compress to make the indexed dimension the last one
    inp_compressed = dim_compress(inp, dim)
    M = inp_compressed.numel() // N

    out = torch.empty_like(inp)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    value_cast = value.to(inp.dtype) if isinstance(value, (int, float)) else value
    index_fill_kernel[grid](
        inp_compressed, out, M, N, index_mask, value_cast
    )

    return out


def index_fill_(inp, dim, index, value):
    logger.debug("GEMS INDEX FILL_")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)

    dim = dim % inp.ndim

    # Create a boolean mask of size N (dim_size) with True at index positions
    N = inp.size(dim)
    index_mask = torch.zeros(N, dtype=torch.bool, device=inp.device)
    index_mask[index] = True

    # Use dim_compress to make the indexed dimension the last one
    inp_compressed = dim_compress(inp, dim)
    M = inp_compressed.numel() // N

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    value_cast = value.to(inp.dtype) if isinstance(value, (int, float)) else value
    index_fill_kernel[grid](
        inp_compressed, inp, M, N, index_mask, value_cast
    )

    return inp