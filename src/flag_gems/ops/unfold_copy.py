import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics({"BLOCK_SIZE": lambda args: 512})
@triton.jit
def _unfold_copy_kernel_3d(
    inp,
    out,
    numel_out,
    dim,
    size,
    step,
    L,
    # Input shape and stride (3 dims)
    inp_shape0,
    inp_shape1,
    inp_shape2,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    # Output shape and stride (4 dims)
    out_shape0,
    out_shape1,
    out_shape2,
    out_shape3,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    unfold_copy kernel for 3D input tensors.

    Input shape: [inp_shape0, inp_shape1, inp_shape2]
    Output shape varies by dim:
    - dim=0: [L, d2, d1, size] with L = (d0-size)/step+1
    - dim=1: [d0, L, d2, size] with L = (d1-size)/step+1
    - dim=2: [d0, d1, L, size] with L = (d2-size)/step+1

    For output index [w, b, i, k] (where w=window, b,d1=other dims, i=d2, k=window_pos):
    - dim=0: output[w, i, b, k] = input[w+k, b, i]
    - dim=1: output[b, w, i, k] = input[b, w+k, i]
    - dim=2: output[b, i, w, k] = input[b, i, w+k]
    """
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel_out

    if dim == 0:
        # Unfold dim 0: output [L, d1, d2, size]
        # output[w, b, i, k] = input[w+k, b, i]
        # out_stride: [d1*d2*size, d2*size, size, 1]
        # Decompose: k, then i (innermost from d2), then b (from d1), then w
        k = offs % size
        remaining = offs // size
        i = remaining % out_shape2  # out_shape2 = d2 (innermost, stride=1)
        remaining = remaining // out_shape2
        b = remaining % out_shape1  # out_shape1 = d1 (stride=d2*size=8)
        remaining = remaining // out_shape1
        w = remaining % out_shape0  # out_shape0 = L

        out_offset = w * out_stride0 + b * out_stride1 + i * out_stride2 + k * out_stride3

        # Input: [w+k, b, i] -> actually [w*step + k, b, i]
        inp_wk = w * step + k
        valid = (inp_wk < inp_shape0) & (b < inp_shape1) & (i < inp_shape2)
        inp_offset = inp_wk * inp_stride0 + b * inp_stride1 + i * inp_stride2

    elif dim == 1:
        # Unfold dim 1: output [d0, L, d2, size]
        # output[b, w, i, k] = input[b, w+k, i]
        # out_stride: [L*d2*size, d2*size, size, 1]
        k = offs % size
        remaining = offs // size
        i = remaining % out_shape2  # out_shape2 = d2
        remaining = remaining // out_shape2
        w = remaining % out_shape1  # out_shape1 = L
        remaining = remaining // out_shape1
        b = remaining % out_shape0  # out_shape0 = d0

        out_offset = b * out_stride0 + w * out_stride1 + i * out_stride2 + k * out_stride3

        # Input: [b, w+k, i] -> actually [b, w*step + k, i]
        inp_b = b
        inp_wk = w * step + k
        valid = (inp_b < inp_shape0) & (inp_wk < inp_shape1) & (i < inp_shape2)
        inp_offset = inp_b * inp_stride0 + inp_wk * inp_stride1 + i * inp_stride2

    else:  # dim == 2
        # Unfold dim 2: output [d0, d1, L, size]
        # output[b, i, w, k] = input[b, i, w+k]
        # out_stride: [d1*L*size, L*size, size, 1]
        k = offs % size
        remaining = offs // size
        w = remaining % out_shape2  # out_shape2 = L
        remaining = remaining // out_shape2
        i = remaining % out_shape1  # out_shape1 = d1
        remaining = remaining // out_shape1
        b = remaining % out_shape0  # out_shape0 = d0

        out_offset = b * out_stride0 + i * out_stride1 + w * out_stride2 + k * out_stride3

        # Input: [b, i, w+k] -> actually [b, i, w*step + k]
        inp_b = b
        inp_i = i
        inp_wk = w * step + k
        valid = (inp_b < inp_shape0) & (inp_i < inp_shape1) & (inp_wk < inp_shape2)
        inp_offset = inp_b * inp_stride0 + inp_i * inp_stride1 + inp_wk * inp_stride2

    vals = tl.load(inp + inp_offset, mask=mask & valid, other=0.0)
    tl.store(out + out_offset, vals, mask=mask & valid)


def unfold_copy(inp: torch.Tensor, dim: int, size: int, step: int) -> torch.Tensor:
    logger.debug("GEMS UNFOLD_COPY")

    if step <= 0:
        raise ValueError("step must be > 0")
    if size <= 0:
        raise ValueError("size must be > 0")
    if dim < -inp.ndim or dim >= inp.ndim:
        raise ValueError(
            f"dimension out of range (expected to be in range of [{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )

    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    ndim = inp.ndim

    if ndim != 3:
        raise NotImplementedError(
            f"unfold_copy only supports 3D input tensors for now, got {ndim}D"
        )

    d = inp_shape[dim]
    L = (d - size) // step + 1
    if L <= 0:
        # Empty output if size > dimension size
        out_shape = inp_shape[:dim] + [0] + inp_shape[dim + 1 :] + [size]
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Build output shape: [d0, d1, ..., d_{dim-1}, L, d_{dim+1}, ..., dn-1, size]
    out_shape = inp_shape[:dim] + [L] + inp_shape[dim + 1 :] + [size]

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if out.numel() == 0:
        return out

    numel_out = out.numel()
    BLOCK = 512
    grid = lambda meta: (triton.cdiv(numel_out, meta["BLOCK_SIZE"]),)

    _unfold_copy_kernel_3d[grid](
        inp,
        out,
        numel_out,
        dim,
        size,
        step,
        L,
        inp.shape[0],
        inp.shape[1],
        inp.shape[2],
        inp.stride()[0],
        inp.stride()[1],
        inp.stride()[2],
        out.shape[0],
        out.shape[1],
        out.shape[2],
        out.shape[3],
        out.stride()[0],
        out.stride()[1],
        out.stride()[2],
        out.stride()[3],
        BLOCK_SIZE=BLOCK,
    )

    return out