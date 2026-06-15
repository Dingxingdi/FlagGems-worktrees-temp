import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["N"])
def diff_kernel_1d(inp, out, N, dim_stride: tl.constexpr, dtype: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Fast path for 1D tensors or when dim is the last dimension"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    current_val = tl.load(inp + offsets * dim_stride)
    next_val = tl.load(inp + (offsets + 1) * dim_stride)

    if dtype == tl.float16:
        current_val = current_val.to(tl.float32)
        next_val = next_val.to(tl.float32)
    elif dtype == tl.bfloat16:
        current_val = current_val.to(tl.float32)
        next_val = next_val.to(tl.float32)

    result = next_val - current_val

    if dtype == tl.float16 or dtype == tl.bfloat16:
        result = result.to(dtype)

    tl.store(out + offsets, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["output_numel"])
def diff_kernel_2d(
    inp,
    out,
    output_numel,
    in_shape0, in_shape1, in_stride0, in_stride1,
    out_shape0, out_shape1, out_stride0, out_stride1,
    dim_idx: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """2D diff kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel

    # Convert flat output index to 2D coordinates
    idx = offsets
    o0 = idx % out_shape0
    o1 = idx // out_shape0

    # Compute output flat offset
    out_offset = o0 * out_stride0 + o1 * out_stride1

    # Compute input offsets based on dim_idx
    if dim_idx == 0:
        offset_in0 = o0 * in_stride0 + o1 * in_stride1
        offset_in1 = (o0 + 1) * in_stride0 + o1 * in_stride1
    else:  # dim_idx == 1
        offset_in0 = o0 * in_stride0 + o1 * in_stride1
        offset_in1 = o0 * in_stride0 + (o1 + 1) * in_stride1

    current_val = tl.load(inp + offset_in0)
    next_val = tl.load(inp + offset_in1)

    if dtype == tl.float16:
        current_val = current_val.to(tl.float32)
        next_val = next_val.to(tl.float32)
    elif dtype == tl.bfloat16:
        current_val = current_val.to(tl.float32)
        next_val = next_val.to(tl.float32)

    result = next_val - current_val

    if dtype == tl.float16 or dtype == tl.bfloat16:
        result = result.to(dtype)

    tl.store(out + out_offset, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["output_numel"])
def diff_kernel_3d(
    inp,
    out,
    output_numel,
    in_shape0, in_shape1, in_shape2,
    in_stride0, in_stride1, in_stride2,
    out_shape0, out_shape1, out_shape2,
    out_stride0, out_stride1, out_stride2,
    dim_idx: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """3D diff kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel

    idx = offsets
    idx_div = idx // out_shape0
    o0 = idx - idx_div * out_shape0
    o1 = idx_div % out_shape1
    o2 = idx_div // out_shape1

    out_offset = o0 * out_stride0 + o1 * out_stride1 + o2 * out_stride2

    if dim_idx == 0:
        offset_in0 = o0 * in_stride0 + o1 * in_stride1 + o2 * in_stride2
        offset_in1 = (o0 + 1) * in_stride0 + o1 * in_stride1 + o2 * in_stride2
    elif dim_idx == 1:
        offset_in0 = o0 * in_stride0 + o1 * in_stride1 + o2 * in_stride2
        offset_in1 = o0 * in_stride0 + (o1 + 1) * in_stride1 + o2 * in_stride2
    else:  # dim_idx == 2
        offset_in0 = o0 * in_stride0 + o1 * in_stride1 + o2 * in_stride2
        offset_in1 = o0 * in_stride0 + o1 * in_stride1 + (o2 + 1) * in_stride2

    current_val = tl.load(inp + offset_in0)
    next_val = tl.load(inp + offset_in1)

    if dtype == tl.float16:
        current_val = current_val.to(tl.float32)
        next_val = next_val.to(tl.float32)
    elif dtype == tl.bfloat16:
        current_val = current_val.to(tl.float32)
        next_val = next_val.to(tl.float32)

    result = next_val - current_val

    if dtype == tl.float16 or dtype == tl.bfloat16:
        result = result.to(dtype)

    tl.store(out + out_offset, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["output_numel"])
def diff_kernel_4d(
    inp,
    out,
    output_numel,
    in_shape0, in_shape1, in_shape2, in_shape3,
    in_stride0, in_stride1, in_stride2, in_stride3,
    out_shape0, out_shape1, out_shape2, out_shape3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    dim_idx: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """4D diff kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel

    idx = offsets
    idx_div = idx // out_shape0
    o0 = idx - idx_div * out_shape0
    idx_div2 = idx_div // out_shape1
    o1 = idx_div - idx_div2 * out_shape1
    o2 = idx_div2 % out_shape2
    o3 = idx_div2 // out_shape2

    out_offset = o0 * out_stride0 + o1 * out_stride1 + o2 * out_stride2 + o3 * out_stride3

    if dim_idx == 0:
        offset_in0 = o0 * in_stride0 + o1 * in_stride1 + o2 * in_stride2 + o3 * in_stride3
        offset_in1 = (o0 + 1) * in_stride0 + o1 * in_stride1 + o2 * in_stride2 + o3 * in_stride3
    elif dim_idx == 1:
        offset_in0 = o0 * in_stride0 + o1 * in_stride1 + o2 * in_stride2 + o3 * in_stride3
        offset_in1 = o0 * in_stride0 + (o1 + 1) * in_stride1 + o2 * in_stride2 + o3 * in_stride3
    elif dim_idx == 2:
        offset_in0 = o0 * in_stride0 + o1 * in_stride1 + o2 * in_stride2 + o3 * in_stride3
        offset_in1 = o0 * in_stride0 + o1 * in_stride1 + (o2 + 1) * in_stride2 + o3 * in_stride3
    else:  # dim_idx == 3
        offset_in0 = o0 * in_stride0 + o1 * in_stride1 + o2 * in_stride2 + o3 * in_stride3
        offset_in1 = o0 * in_stride0 + o1 * in_stride1 + o2 * in_stride2 + (o3 + 1) * in_stride3

    current_val = tl.load(inp + offset_in0)
    next_val = tl.load(inp + offset_in1)

    if dtype == tl.float16:
        current_val = current_val.to(tl.float32)
        next_val = next_val.to(tl.float32)
    elif dtype == tl.bfloat16:
        current_val = current_val.to(tl.float32)
        next_val = next_val.to(tl.float32)

    result = next_val - current_val

    if dtype == tl.float16 or dtype == tl.bfloat16:
        result = result.to(dtype)

    tl.store(out + out_offset, result, mask=mask)


def _get_triton_dtype(dtype):
    """Convert torch dtype to triton dtype"""
    dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        torch.int16: tl.int16,
        torch.int8: tl.int8,
        torch.uint8: tl.uint8,
        torch.bool: tl.int1,
    }
    return dtype_map.get(dtype, tl.float32)


def diff(inp, n=1, dim=-1, prepend=None, append=None):
    logger.debug("GEMS DIFF")

    # Handle n > 1 recursively
    if n > 1:
        out = diff(inp, n=1, dim=dim, prepend=prepend, append=append)
        return diff(out, n=n - 1, dim=dim)

    dim = dim % inp.ndim

    # Handle prepend/append by concatenating
    if prepend is not None or append is not None:
        tensors_to_cat = []
        if prepend is not None:
            tensors_to_cat.append(prepend)
        tensors_to_cat.append(inp)
        if append is not None:
            tensors_to_cat.append(append)
        inp = torch.cat(tensors_to_cat, dim=dim)

    dim_size = inp.size(dim)

    output_shape = list(inp.shape)
    output_shape[dim] = max(0, dim_size - 1)
    output_numel = 1
    for s in output_shape:
        output_numel *= s

    if output_numel == 0:
        # PyTorch returns empty tensor when input has size 1 along dim
        return torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    dtype = _get_triton_dtype(inp.dtype)
    BLOCK_SIZE = 1024
    num_warps = 8

    # Fast path for 1D tensors only (2D with dim=-1 has different output stride)
    if inp.ndim == 1:
        dim_stride = inp.stride(dim)
        grid = (triton.cdiv(output_numel, BLOCK_SIZE),)
        diff_kernel_1d[grid](
            inp,
            output,
            output_numel,
            dim_stride,
            dtype,
            BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        # Multi-dimensional case (dim is not the last dimension)
        ndim = inp.ndim
        out_stride = output.stride()
        in_stride = inp.stride()

        grid = (triton.cdiv(output_numel, BLOCK_SIZE),)

        if ndim == 2:
            diff_kernel_2d[grid](
                inp,
                output,
                output_numel,
                inp.shape[0], inp.shape[1],
                in_stride[0], in_stride[1],
                output.shape[0], output.shape[1],
                out_stride[0], out_stride[1],
                dim,
                dtype,
                BLOCK_SIZE,
                num_warps=num_warps,
            )
        elif ndim == 3:
            diff_kernel_3d[grid](
                inp,
                output,
                output_numel,
                inp.shape[0], inp.shape[1], inp.shape[2],
                in_stride[0], in_stride[1], in_stride[2],
                output.shape[0], output.shape[1], output.shape[2],
                out_stride[0], out_stride[1], out_stride[2],
                dim,
                dtype,
                BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:  # ndim == 4
            diff_kernel_4d[grid](
                inp,
                output,
                output_numel,
                inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3],
                in_stride[0], in_stride[1], in_stride[2], in_stride[3],
                output.shape[0], output.shape[1], output.shape[2], output.shape[3],
                out_stride[0], out_stride[1], out_stride[2], out_stride[3],
                dim,
                dtype,
                BLOCK_SIZE,
                num_warps=num_warps,
            )

    return output


def diff_(inp, n=1, dim=-1, prepend=None, append=None):
    """In-place version of diff"""
    logger.debug("GEMS DIFF_")
    result = diff(inp, n=n, dim=dim, prepend=prepend, append=append)
    inp.copy_(result)
    return inp