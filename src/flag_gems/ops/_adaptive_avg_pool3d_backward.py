import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def adaptive_avg_pool3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    # Strides for grad_output
    stride_go_n,
    stride_go_c,
    stride_go_d,
    stride_go_h,
    stride_go_w,
    # Strides for grad_input
    stride_gi_n,
    stride_gi_c,
    stride_gi_d,
    stride_gi_h,
    stride_gi_w,
    # Block size for tiling over input spatial dimensions
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Unravel pid -> (n, c, d_o, h_o, w_o) - each thread handles one output position
    W_out_i64 = tl.full((), W_out, tl.int64)
    H_out_i64 = tl.full((), H_out, tl.int64)
    D_out_i64 = tl.full((), D_out, tl.int64)
    C_i64 = tl.full((), C, tl.int64)

    idx = tl.cast(pid, tl.int64)
    w_o = idx % W_out_i64
    idx = idx // W_out_i64
    h_o = idx % H_out_i64
    idx = idx // H_out_i64
    d_o = idx % D_out_i64
    idx = idx // D_out_i64
    c = idx % C_i64
    n = idx // C_i64

    # Compute the input region that this output position covers
    D_in_i64 = tl.full((), D_in, tl.int64)
    H_in_i64 = tl.full((), H_in, tl.int64)
    W_in_i64 = tl.full((), W_in, tl.int64)
    D_out_i64 = tl.full((), D_out, tl.int64)
    H_out_i64 = tl.full((), H_out, tl.int64)
    W_out_i64 = tl.full((), W_out, tl.int64)

    # d0 = (d_o * D_in) // D_out
    # d1 = ((d_o + 1) * D_in + D_out - 1) // D_out
    d0 = (d_o * D_in_i64) // D_out_i64
    d1 = ((d_o + 1) * D_in_i64 + D_out_i64 - 1) // D_out_i64
    dd = d1 - d0

    h0 = (h_o * H_in_i64) // H_out_i64
    h1 = ((h_o + 1) * H_in_i64 + H_out_i64 - 1) // H_out_i64
    hh = h1 - h0

    w0 = (w_o * W_in_i64) // W_out_i64
    w1 = ((w_o + 1) * W_in_i64 + W_out_i64 - 1) // W_out_i64
    ww = w1 - w0

    count = dd * hh * ww
    count_f = tl.cast(count, tl.float32)

    # Load grad_output at output position
    grad_out_idx = (
        n * stride_go_n
        + c * stride_go_c
        + d_o * stride_go_d
        + h_o * stride_go_h
        + w_o * stride_go_w
    )
    grad_out_val = tl.load(grad_output_ptr + grad_out_idx)
    grad_val = tl.cast(grad_out_val, tl.float32) / count_f

    # Base pointers for input
    base_nc = n * stride_gi_n + c * stride_gi_c

    # Tiled iteration over input region
    for d_i_base in range(d0, d1, BLOCK_D):
        for h_i_base in range(h0, h1, BLOCK_H):
            for w_i_base in range(w0, w1, BLOCK_W):
                # Generate block offsets
                d_offsets = d_i_base + tl.arange(0, BLOCK_D)
                h_offsets = h_i_base + tl.arange(0, BLOCK_H)
                w_offsets = w_i_base + tl.arange(0, BLOCK_W)

                # Create masks for valid elements in this block
                d_mask = d_offsets < d1
                h_mask = h_offsets < h1
                w_mask = w_offsets < w1
                mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

                # Compute flat indices for grad_input
                offsets = (
                    base_nc
                    + d_offsets[:, None, None] * stride_gi_d
                    + h_offsets[None, :, None] * stride_gi_h
                    + w_offsets[None, None, :] * stride_gi_w
                )

                # Store gradient to all elements in the block
                tl.store(
                    grad_input_ptr + offsets,
                    tl.cast(grad_val, grad_input_ptr.type.element_ty),
                    mask=mask,
                )


def _adaptive_avg_pool3d_backward(grad_output: torch.Tensor, input: torch.Tensor):
    logger.debug("GEMS _ADAPTIVE_AVG_POOL3D_BACKWARD")

    N, C, D_in, H_in, W_in = input.shape
    D_out, H_out, W_out = grad_output.shape[-3], grad_output.shape[-2], grad_output.shape[-1]

    grad_input = torch.empty_like(input, dtype=torch.float32)

    if grad_output.numel() == 0 or input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    total = N * C * D_out * H_out * W_out
    if total == 0:
        return grad_input.to(grad_output.dtype)

    # Use small block sizes for better cache utilization
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 4

    grid = (total,)

    adaptive_avg_pool3d_backward_kernel[grid](
        grad_output,
        grad_input,
        N,
        C,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_output.stride(4),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_input.stride(4),
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
    )

    return grad_input.to(grad_output.dtype)