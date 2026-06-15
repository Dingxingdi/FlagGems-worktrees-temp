import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems import runtime
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def index_reduce_prod_kernel_1d(
    inp,
    index,
    src,
    out,
    n_indices,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Index reduce kernel for 1D case with prod reduction."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_indices

    # Load source value
    src_val = tl.load(src + offsets, mask=mask, other=1.0)

    # Load index
    idx = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)

    # Load current value from input
    cur_val = tl.load(inp + idx, mask=(idx >= 0) & (idx < dim_size), other=1.0)

    # Apply prod reduction
    new_val = cur_val * src_val

    # Store back
    tl.store(inp + idx, new_val, mask=(idx >= 0) & (idx < dim_size))


@triton.jit
def index_reduce_prod_kernel_2d(
    inp,
    index,
    src,
    out,
    n_elements,
    n_indices,
    src_size_dim,
    dim_size,
    src_stride_0,
    src_stride_1,
    inp_stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    """Index reduce kernel for 2D case with prod reduction (dim=0)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Process all src elements

    # Calculate source element's position
    src_offset_0 = offsets // src_stride_0
    src_offset_1 = offsets % src_stride_0

    # Load source value
    src_val = tl.load(src + offsets, mask=mask, other=1.0)

    # Load index - use src_offset_0 to get the correct index for this element
    # Since index has size = src_size_dim (the size along the reduction dim)
    idx = tl.load(index + src_offset_0, mask=(src_offset_0 < n_indices), other=0).to(tl.int64)

    # Calculate input offset
    # For dim=0: self[idx, src_offset_1]
    inp_offset = idx * inp_stride_0 + src_offset_1

    # Load current value from input
    cur_val = tl.load(inp + inp_offset, mask=(idx >= 0) & (idx < dim_size), other=1.0)

    # Apply prod reduction
    new_val = cur_val * src_val

    # Store back
    tl.store(inp + inp_offset, new_val, mask=(idx >= 0) & (idx < dim_size))


@triton.jit
def index_reduce_amax_kernel_1d(
    inp,
    index,
    src,
    out,
    n_indices,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Index reduce kernel for 1D case with amax reduction."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_indices

    # Load source value
    src_val = tl.load(src + offsets, mask=mask, other=-float('inf'))

    # Load index
    idx = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)

    # Load current value from input
    cur_val = tl.load(inp + idx, mask=(idx >= 0) & (idx < dim_size), other=-float('inf'))

    # Apply amax reduction
    new_val = tl.maximum(cur_val, src_val)

    # Store back
    tl.store(inp + idx, new_val, mask=(idx >= 0) & (idx < dim_size))


@triton.jit
def index_reduce_amin_kernel_1d(
    inp,
    index,
    src,
    out,
    n_indices,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Index reduce kernel for 1D case with amin reduction."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_indices

    # Load source value
    src_val = tl.load(src + offsets, mask=mask, other=float('inf'))

    # Load index
    idx = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)

    # Load current value from input
    cur_val = tl.load(inp + idx, mask=(idx >= 0) & (idx < dim_size), other=float('inf'))

    # Apply amin reduction
    new_val = tl.minimum(cur_val, src_val)

    # Store back
    tl.store(inp + idx, new_val, mask=(idx >= 0) & (idx < dim_size))


@triton.jit
def index_reduce_sum_kernel_1d(
    inp,
    index,
    src,
    out,
    n_indices,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Index reduce kernel for 1D case with sum reduction (used for mean)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_indices

    # Load source value
    src_val = tl.load(src + offsets, mask=mask, other=0.0)

    # Load index
    idx = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)

    # Load current value from input
    cur_val = tl.load(inp + idx, mask=(idx >= 0) & (idx < dim_size), other=0.0)

    # Apply sum reduction
    new_val = cur_val + src_val

    # Store back
    tl.store(inp + idx, new_val, mask=(idx >= 0) & (idx < dim_size))


def _index_reduce_1d(inp, dim, index, src, reduce, include_self):
    """Handle 1D input case."""
    n_indices = index.numel()
    dim_size = inp.size(dim)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_indices, BLOCK_SIZE),)

    if reduce == "prod":
        index_reduce_prod_kernel_1d[grid](
            inp, index, src, inp, n_indices, dim_size, BLOCK_SIZE=BLOCK_SIZE
        )
    elif reduce == "amax":
        index_reduce_amax_kernel_1d[grid](
            inp, index, src, inp, n_indices, dim_size, BLOCK_SIZE=BLOCK_SIZE
        )
    elif reduce == "amin":
        index_reduce_amin_kernel_1d[grid](
            inp, index, src, inp, n_indices, dim_size, BLOCK_SIZE=BLOCK_SIZE
        )
    elif reduce == "mean":
        index_reduce_sum_kernel_1d[grid](
            inp, index, src, inp, n_indices, dim_size, BLOCK_SIZE=BLOCK_SIZE
        )
    return inp


def _index_reduce_2d(inp, dim, index, src, reduce, include_self):
    """Handle 2D input case (dim=0 only)."""
    n_indices = index.numel()
    src_size_dim = src.size(dim)
    dim_size = inp.size(dim)
    BLOCK_SIZE = 128

    if reduce == "prod" and dim == 0:
        src_stride_0 = src.stride(0)
        src_stride_1 = src.stride(1)
        inp_stride_0 = inp.stride(0)
        n_elements = src.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        index_reduce_prod_kernel_2d[grid](
            inp, index, src, inp, n_elements, n_indices, src_size_dim, dim_size,
            src_stride_0, src_stride_1, inp_stride_0, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Fall back to PyTorch for unsupported cases
        return torch.index_reduce(inp, dim, index, src, reduce, include_self=include_self)
    return inp


def index_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS INDEX REDUCE")

    # Validate inputs
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(dim), (
        "The dimth dimension of source must have the same size as the length of index"
    )
    assert inp.ndim == src.ndim, (
        "Self and source should have the same number of dimensions"
    )
    assert reduce in ["prod", "mean", "amax", "amin"], (
        f"reduce must be one of prod, mean, amax, amin, got {reduce}"
    )

    dim %= inp.ndim
    out = inp.clone()

    if inp.ndim == 1:
        return _index_reduce_1d(out, dim, index, src, reduce, include_self)
    elif inp.ndim == 2 and dim == 0:
        return _index_reduce_2d(out, dim, index, src, reduce, include_self)
    else:
        # Fall back to PyTorch for unsupported cases
        return torch.index_reduce(out, dim, index, src, reduce, include_self=include_self)


def index_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS INDEX REDUCE_")

    # Validate inputs
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(dim), (
        "The dimth dimension of source must have the same size as the length of index"
    )
    assert inp.ndim == src.ndim, (
        "Self and source should have the same number of dimensions"
    )
    assert reduce in ["prod", "mean", "amax", "amin"], (
        f"reduce must be one of prod, mean, amax, amin, got {reduce}"
    )

    dim %= inp.ndim

    if inp.ndim == 1:
        return _index_reduce_1d(inp, dim, index, src, reduce, include_self)
    elif inp.ndim == 2 and dim == 0:
        return _index_reduce_2d(inp, dim, index, src, reduce, include_self)
    else:
        # Fall back to PyTorch for unsupported cases
        inp.index_reduce_(dim, index, src, reduce, include_self=include_self)
        return inp