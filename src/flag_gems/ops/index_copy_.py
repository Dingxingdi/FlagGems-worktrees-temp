import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def index_copy_kernel_1d(
    inp,
    out,
    index,
    src,
    N,
    index_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # For 1D, dim must be 0
    target_idx = tl.load(index + offsets, mask=mask, other=0)
    target_idx = target_idx.to(tl.int64)

    val = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(inp + target_idx, val, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def index_copy_kernel_2d(
    inp,
    out,
    index,
    src,
    N,
    dim,
    inp_stride_dim,
    src_stride_dim,
    inp_shape_dim,
    src_shape_dim,
    index_numel,
    src_stride_0,
    src_stride_1,
    src_shape_0,
    src_shape_1,
    inp_stride_0,
    inp_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Compute 2D indices
    idx_0 = offsets % src_shape_0
    idx_1 = offsets // src_shape_0

    # Source offset
    src_offset = idx_0 * src_stride_0 + idx_1 * src_stride_1

    # Get target index
    if dim == 0:
        dim_idx = idx_0
        other_idx = idx_1
        target_idx = tl.load(index + dim_idx, mask=dim_idx < index_numel, other=0)
        target_idx = target_idx.to(tl.int64)
        inp_offset = target_idx * inp_stride_0 + other_idx * inp_stride_1
    else:
        dim_idx = idx_1
        other_idx = idx_0
        target_idx = tl.load(index + dim_idx, mask=dim_idx < index_numel, other=0)
        target_idx = target_idx.to(tl.int64)
        inp_offset = other_idx * inp_stride_0 + target_idx * inp_stride_1

    # Load and store
    val = tl.load(src + src_offset, mask=mask, other=0.0)
    tl.store(inp + inp_offset, val, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def index_copy_kernel_3d(
    inp,
    out,
    index,
    src,
    N,
    dim,
    inp_stride_dim,
    src_stride_dim,
    inp_shape_dim,
    src_shape_dim,
    index_numel,
    src_stride_0,
    src_stride_1,
    src_stride_2,
    src_shape_0,
    src_shape_1,
    src_shape_2,
    inp_stride_0,
    inp_stride_1,
    inp_stride_2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Compute 3D indices
    idx_0 = offsets % src_shape_0
    remainder = offsets // src_shape_0
    idx_1 = remainder % src_shape_1
    idx_2 = remainder // src_shape_1

    # Source offset
    src_offset = idx_0 * src_stride_0 + idx_1 * src_stride_1 + idx_2 * src_stride_2

    # Get target index and compute inp offset based on dim
    if dim == 0:
        dim_idx = idx_0
        other_0 = idx_1
        other_1 = idx_2
        target_idx = tl.load(index + dim_idx, mask=dim_idx < index_numel, other=0)
        target_idx = target_idx.to(tl.int64)
        inp_offset = (
            target_idx * inp_stride_0 + other_0 * inp_stride_1 + other_1 * inp_stride_2
        )
    elif dim == 1:
        dim_idx = idx_1
        other_0 = idx_0
        other_1 = idx_2
        target_idx = tl.load(index + dim_idx, mask=dim_idx < index_numel, other=0)
        target_idx = target_idx.to(tl.int64)
        inp_offset = (
            other_0 * inp_stride_0 + target_idx * inp_stride_1 + other_1 * inp_stride_2
        )
    else:
        dim_idx = idx_2
        other_0 = idx_0
        other_1 = idx_1
        target_idx = tl.load(index + dim_idx, mask=dim_idx < index_numel, other=0)
        target_idx = target_idx.to(tl.int64)
        inp_offset = (
            other_0 * inp_stride_0 + other_1 * inp_stride_1 + target_idx * inp_stride_2
        )

    # Load and store
    val = tl.load(src + src_offset, mask=mask, other=0.0)
    tl.store(inp + inp_offset, val, mask=mask)


def index_copy(inp, dim, index, source):
    logger.debug("GEMS INDEX COPY")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim == 1, "Index should have dimension 1"
    assert index.numel() == source.size(dim), (
        "The dimth dimension of source must have the same size as the length of index"
    )
    assert inp.ndim == source.ndim, (
        "Self and source should have the same number of dimensions"
    )

    dim %= inp.ndim
    index = index.to(torch.int64)
    source = source.to(inp.dtype)

    out = inp.clone()
    N = source.numel()
    index_numel = index.numel()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    rank = inp.ndim
    if rank == 1:
        index_copy_kernel_1d[grid](
            out,
            out,
            index,
            source,
            N,
            index_numel,
        )
    elif rank == 2:
        index_copy_kernel_2d[grid](
            out,
            out,
            index,
            source,
            N,
            dim,
            inp.stride(dim),
            source.stride(dim),
            inp.size(dim),
            source.size(dim),
            index_numel,
            source.stride(0),
            source.stride(1),
            source.size(0),
            source.size(1),
            inp.stride(0),
            inp.stride(1),
        )
    elif rank == 3:
        index_copy_kernel_3d[grid](
            out,
            out,
            index,
            source,
            N,
            dim,
            inp.stride(dim),
            source.stride(dim),
            inp.size(dim),
            source.size(dim),
            index_numel,
            source.stride(0),
            source.stride(1),
            source.stride(2),
            source.size(0),
            source.size(1),
            source.size(2),
            inp.stride(0),
            inp.stride(1),
            inp.stride(2),
        )
    else:
        # For other ranks, fall back to torch implementation
        out = torch.index_copy(inp, dim, index, source)

    return out


def index_copy_(inp, dim, index, source):
    logger.debug("GEMS INDEX COPY_")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim == 1, "Index should have dimension 1"
    assert index.numel() == source.size(dim), (
        "The dimth dimension of source must have the same size as the length of index"
    )
    assert inp.ndim == source.ndim, (
        "Self and source should have the same number of dimensions"
    )

    dim %= inp.ndim
    index = index.to(torch.int64)
    source = source.to(inp.dtype)

    N = source.numel()
    index_numel = index.numel()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    rank = inp.ndim
    if rank == 1:
        index_copy_kernel_1d[grid](
            inp,
            inp,
            index,
            source,
            N,
            index_numel,
        )
    elif rank == 2:
        index_copy_kernel_2d[grid](
            inp,
            inp,
            index,
            source,
            N,
            dim,
            inp.stride(dim),
            source.stride(dim),
            inp.size(dim),
            source.size(dim),
            index_numel,
            source.stride(0),
            source.stride(1),
            source.size(0),
            source.size(1),
            inp.stride(0),
            inp.stride(1),
        )
    elif rank == 3:
        index_copy_kernel_3d[grid](
            inp,
            inp,
            index,
            source,
            N,
            dim,
            inp.stride(dim),
            source.stride(dim),
            inp.size(dim),
            source.size(dim),
            index_numel,
            source.stride(0),
            source.stride(1),
            source.stride(2),
            source.size(0),
            source.size(1),
            source.size(2),
            inp.stride(0),
            inp.stride(1),
            inp.stride(2),
        )
    else:
        # For other ranks, fall back to torch implementation
        torch.index_copy_(inp, dim, index, source)

    return inp