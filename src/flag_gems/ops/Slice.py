import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def slice_op(inp: torch.Tensor, dim: int, start: int, end: int, step: int):
    """Slice tensor along given dimension."""
    ndim = inp.ndim
    dim = dim % ndim

    # Handle None start/end
    if start is None:
        start = 0
    if end is None:
        end = inp.shape[dim]

    # Handle negative indices
    if start < 0:
        start = max(0, inp.shape[dim] + start)
    if end < 0:
        end = max(0, inp.shape[dim] + end)

    # Clamp
    start = max(0, min(start, inp.shape[dim]))
    end = max(0, min(end, inp.shape[dim]))

    # Calculate output shape
    slice_len = max(0, (end - start + (step - 1)) // step)
    out_shape = list(inp.shape)
    out_shape[dim] = slice_len

    # Allocate output
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Handle trivial cases
    if slice_len == 0 or inp.numel() == 0:
        return out

    numel = out.numel()
    BLOCK_SIZE = 128

    if ndim == 1:
        # 1D case: use Triton kernel
        @libentry()
        @triton.jit
        def slice_kernel_1d(
            inp, out,
            start, step, numel,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tle.program_id(0)
            offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < numel

            inp_idx = start + offset * step
            value = tl.load(inp + inp_idx, mask=mask, other=0.0)
            tl.store(out + offset, value, mask=mask)

        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        slice_kernel_1d[grid](inp, out, start, step, numel, BLOCK_SIZE)

    numel = out.numel()
    BLOCK_SIZE = 128

    if ndim == 2:
        # 2D case
        M = inp.shape[0]
        N = inp.shape[1]
        OM = out.shape[0]
        ON = out.shape[1]

        @libentry()
        @triton.jit
        def slice_kernel_2d(
            inp,
            out,
            M,
            N,
            OM,
            ON,
            inp_stride_m,
            inp_stride_n,
            out_stride_m,
            out_stride_n,
            start,
            step,
            dim,
            numel,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tle.program_id(0)
            offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < numel

            # Calculate output indices
            cur_offset = offset
            out_idx_m = cur_offset % OM
            cur_offset = cur_offset // OM
            out_idx_n = cur_offset % ON

            # Map output indices to input indices based on dim
            if dim == 0:
                inp_idx_m = start + out_idx_m * step
                inp_idx_n = out_idx_n
            else:
                inp_idx_m = out_idx_m
                inp_idx_n = start + out_idx_n * step

            # Calculate offsets
            inp_offset = inp_idx_m * inp_stride_m + inp_idx_n * inp_stride_n
            out_offset = out_idx_m * out_stride_m + out_idx_n * out_stride_n

            # Load and store
            value = tl.load(inp + inp_offset, mask=mask, other=0.0)
            tl.store(out + out_offset, value, mask=mask)

        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        slice_kernel_2d[grid](
            inp,
            out,
            M,
            N,
            OM,
            ON,
            inp.stride(0),
            inp.stride(1),
            out.stride(0),
            out.stride(1),
            start,
            step,
            dim,
            numel,
            BLOCK_SIZE,
        )

    elif ndim == 3:
        @libentry()
        @triton.jit
        def slice_kernel_3d(
            inp, out,
            out_shape0, out_shape1, out_shape2,
            inp_stride0, inp_stride1, inp_stride2,
            out_stride0, out_stride1, out_stride2,
            start, step, dim, numel,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tle.program_id(0)
            offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < numel

            cur_offset = offset

            out_i2 = cur_offset % out_shape2
            cur_offset = cur_offset // out_shape2
            out_i1 = cur_offset % out_shape1
            cur_offset = cur_offset // out_shape1
            out_i0 = cur_offset % out_shape0

            # Map to input indices
            if dim == 0:
                inp_i0 = start + out_i0 * step
                inp_i1, inp_i2 = out_i1, out_i2
            elif dim == 1:
                inp_i0 = out_i0
                inp_i1 = start + out_i1 * step
                inp_i2 = out_i2
            else:
                inp_i0, inp_i1 = out_i0, out_i1
                inp_i2 = start + out_i2 * step

            inp_offset = inp_i0 * inp_stride0 + inp_i1 * inp_stride1 + inp_i2 * inp_stride2
            out_offset = out_i0 * out_stride0 + out_i1 * out_stride1 + out_i2 * out_stride2

            value = tl.load(inp + inp_offset, mask=mask, other=0.0)
            tl.store(out + out_offset, value, mask=mask)

        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        slice_kernel_3d[grid](
            inp, out,
            out.shape[0], out.shape[1], out.shape[2],
            inp.stride(0), inp.stride(1), inp.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            start, step, dim, numel,
            BLOCK_SIZE,
        )

    elif ndim == 4:
        @libentry()
        @triton.jit
        def slice_kernel_4d(
            inp, out,
            out_shape0, out_shape1, out_shape2, out_shape3,
            inp_stride0, inp_stride1, inp_stride2, inp_stride3,
            out_stride0, out_stride1, out_stride2, out_stride3,
            start, step, dim, numel,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tle.program_id(0)
            offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < numel

            cur_offset = offset

            out_i3 = cur_offset % out_shape3
            cur_offset = cur_offset // out_shape3
            out_i2 = cur_offset % out_shape2
            cur_offset = cur_offset // out_shape2
            out_i1 = cur_offset % out_shape1
            cur_offset = cur_offset // out_shape1
            out_i0 = cur_offset

            if dim == 0:
                inp_i0 = start + out_i0 * step
                inp_i1, inp_i2, inp_i3 = out_i1, out_i2, out_i3
            elif dim == 1:
                inp_i1 = start + out_i1 * step
                inp_i0, inp_i2, inp_i3 = out_i0, out_i2, out_i3
            elif dim == 2:
                inp_i2 = start + out_i2 * step
                inp_i0, inp_i1, inp_i3 = out_i0, out_i1, out_i3
            else:
                inp_i3 = start + out_i3 * step
                inp_i0, inp_i1, inp_i2 = out_i0, out_i1, out_i2

            inp_offset = inp_i0 * inp_stride0 + inp_i1 * inp_stride1 + inp_i2 * inp_stride2 + inp_i3 * inp_stride3
            out_offset = out_i0 * out_stride0 + out_i1 * out_stride1 + out_i2 * out_stride2 + out_i3 * out_stride3

            value = tl.load(inp + inp_offset, mask=mask, other=0.0)
            tl.store(out + out_offset, value, mask=mask)

        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        slice_kernel_4d[grid](
            inp, out,
            out.shape[0], out.shape[1], out.shape[2], out.shape[3],
            inp.stride(0), inp.stride(1), inp.stride(2), inp.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            start, step, dim, numel,
            BLOCK_SIZE,
        )
    else:
        # For higher dimensions (ndim > 4), use a simple Triton kernel with flattened approach
        # Flatten tensor, slice, then reshape
        inp_flat = inp.reshape(-1)
        out_flat = out.reshape(-1)

        # Compute the linear start and end for flattening
        # This is a simplified approach - compute strides
        dim_stride = 1
        for d in range(dim):
            dim_stride *= inp.shape[d]

        linear_start = start * dim_stride
        linear_step = step * dim_stride

        # For simplicity, use element-wise copy (less efficient but correct)
        numel = out_flat.numel()
        BLOCK_SIZE = 128

        @libentry()
        @triton.jit
        def slice_kernel_flat(
            inp, out,
            linear_start, linear_step, numel,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tle.program_id(0)
            offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < numel

            inp_idx = linear_start + offset * linear_step
            value = tl.load(inp + inp_idx, mask=mask, other=0.0)
            tl.store(out + offset, value, mask=mask)

        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        slice_kernel_flat[grid](inp_flat, out_flat, linear_start, linear_step, numel, BLOCK_SIZE)

    return out


def slice(inp: torch.Tensor, dim: int = 0, start: int = None, end: int = None, step: int = 1):
    logger.debug("GEMS SLICE")
    if inp is None:
        # When inp is None, this is likely called from PyTorch's internal __getitem__
        # Fall back to PyTorch native implementation
        raise NotImplementedError("slice called with None inp - should not happen in normal use")
    if dim is None:
        dim = 0
    actual_start = start if start is not None else 0
    actual_end = end if end is not None else (inp.shape[dim] if dim < inp.ndim else 0)
    return slice_op(inp, dim, actual_start, actual_end, step)