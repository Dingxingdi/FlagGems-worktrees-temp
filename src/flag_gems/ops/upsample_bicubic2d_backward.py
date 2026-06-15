import logging
import math
from typing import Sequence

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def cubic_weight(d, a: tl.constexpr):
    ad = tl.abs(d)
    ad2 = ad * ad
    ad3 = ad2 * ad
    w1 = (a + 2.0) * ad3 - (a + 3.0) * ad2 + 1.0
    w2 = a * ad3 - 5.0 * a * ad2 + 8.0 * a * ad - 4.0 * a
    return tl.where(ad <= 1.0, w1, tl.where(ad < 2.0, w2, 0.0))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 128}, num_warps=4),
        triton.Config({"BLOCK_W": 256}, num_warps=4),
        triton.Config({"BLOCK_W": 512}, num_warps=8),
    ],
    key=["W_in"],
)
@triton.jit
def _upsample_bicubic2d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    stride_N,
    stride_C,
    stride_H_in,
    stride_W_in,
    out_stride_N,
    out_stride_C,
    out_stride_H_out,
    out_stride_W_out,
    scale_h,
    scale_w,
    align_corners: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Backward kernel for bicubic 2D upsampling.

    For each input position, compute gradient by summing contributions from
    output positions that depend on it.
    """
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W_in, BLOCK_W)

    pid_w = pid % num_w_blocks
    row_id = pid // num_w_blocks

    # Process input positions
    y_in = row_id % H_in
    nc = row_id // H_in
    c = nc % C
    n = nc // C

    # For backward: compute where this input position maps to in output space
    # This is the inverse of the forward mapping
    fy = y_in * 1.0
    if align_corners:
        out_y = fy / scale_h
    else:
        out_y = (fy + 0.5) / scale_h - 0.5

    y0f = tl.floor(out_y)
    y0 = y0f.to(tl.int32)
    ty = out_y - y0f

    # Get the 4x4 neighborhood in OUTPUT that affects this input
    y_m1 = tl.maximum(0, tl.minimum(H_out - 1, y0 - 1))
    y_0 = tl.maximum(0, tl.minimum(H_out - 1, y0 + 0))
    y_p1 = tl.maximum(0, tl.minimum(H_out - 1, y0 + 1))
    y_p2 = tl.maximum(0, tl.minimum(H_out - 1, y0 + 2))

    a = -0.75
    wy0 = cubic_weight(1.0 + ty, a)
    wy1 = cubic_weight(ty, a)
    wy2 = cubic_weight(1.0 - ty, a)
    wy3 = cubic_weight(2.0 - ty, a)

    n_64 = n.to(tl.int64)
    c_64 = c.to(tl.int64)

    base_ptr = grad_out_ptr + n_64 * out_stride_N + c_64 * out_stride_C

    row_m1_ptr = base_ptr + y_m1.to(tl.int64) * out_stride_H_out
    row_0_ptr = base_ptr + y_0.to(tl.int64) * out_stride_H_out
    row_p1_ptr = base_ptr + y_p1.to(tl.int64) * out_stride_H_out
    row_p2_ptr = base_ptr + y_p2.to(tl.int64) * out_stride_H_out

    x_in = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = x_in < W_in

    fx = x_in.to(tl.float32)
    if align_corners:
        out_x = fx / scale_w
    else:
        out_x = (fx + 0.5) / scale_w - 0.5

    x0f = tl.floor(out_x)
    x0 = x0f.to(tl.int32)
    tx = out_x - x0f

    x_m1 = tl.maximum(0, tl.minimum(W_out - 1, x0 - 1))
    x_0 = tl.maximum(0, tl.minimum(W_out - 1, x0 + 0))
    x_p1 = tl.maximum(0, tl.minimum(W_out - 1, x0 + 1))
    x_p2 = tl.maximum(0, tl.minimum(W_out - 1, x0 + 2))

    wx0 = cubic_weight(1.0 + tx, a)
    wx1 = cubic_weight(tx, a)
    wx2 = cubic_weight(1.0 - tx, a)
    wx3 = cubic_weight(2.0 - tx, a)

    off_x_m1 = x_m1 * out_stride_W_out
    off_x_0 = x_0 * out_stride_W_out
    off_x_p1 = x_p1 * out_stride_W_out
    off_x_p2 = x_p2 * out_stride_W_out

    # Accumulate gradient from 4x4 neighborhood
    v0 = tl.load(row_m1_ptr + off_x_m1, mask=mask).to(tl.float32)
    v1 = tl.load(row_m1_ptr + off_x_0, mask=mask).to(tl.float32)
    v2 = tl.load(row_m1_ptr + off_x_p1, mask=mask).to(tl.float32)
    v3 = tl.load(row_m1_ptr + off_x_p2, mask=mask).to(tl.float32)
    acc = (v0 * wx0 + v1 * wx1 + v2 * wx2 + v3 * wx3) * wy0

    v0 = tl.load(row_0_ptr + off_x_m1, mask=mask).to(tl.float32)
    v1 = tl.load(row_0_ptr + off_x_0, mask=mask).to(tl.float32)
    v2 = tl.load(row_0_ptr + off_x_p1, mask=mask).to(tl.float32)
    v3 = tl.load(row_0_ptr + off_x_p2, mask=mask).to(tl.float32)
    acc += (v0 * wx0 + v1 * wx1 + v2 * wx2 + v3 * wx3) * wy1

    v0 = tl.load(row_p1_ptr + off_x_m1, mask=mask).to(tl.float32)
    v1 = tl.load(row_p1_ptr + off_x_0, mask=mask).to(tl.float32)
    v2 = tl.load(row_p1_ptr + off_x_p1, mask=mask).to(tl.float32)
    v3 = tl.load(row_p1_ptr + off_x_p2, mask=mask).to(tl.float32)
    acc += (v0 * wx0 + v1 * wx1 + v2 * wx2 + v3 * wx3) * wy2

    v0 = tl.load(row_p2_ptr + off_x_m1, mask=mask).to(tl.float32)
    v1 = tl.load(row_p2_ptr + off_x_0, mask=mask).to(tl.float32)
    v2 = tl.load(row_p2_ptr + off_x_p1, mask=mask).to(tl.float32)
    v3 = tl.load(row_p2_ptr + off_x_p2, mask=mask).to(tl.float32)
    acc += (v0 * wx0 + v1 * wx1 + v2 * wx2 + v3 * wx3) * wy3

    # Store to grad_input
    out_offset = (
        n_64 * stride_N
        + c_64 * stride_C
        + y_in.to(tl.int64) * stride_H_in
        + x_in.to(tl.int64) * stride_W_in
    )
    tl.store(grad_in_ptr + out_offset, acc.to(grad_in_ptr.dtype.element_ty), mask=mask)


def upsample_bicubic2d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    align_corners: bool = False,
    scales_h: float | None = None,
    scales_w: float | None = None,
) -> torch.Tensor:
    """
    Backward function for upsample_bicubic2d.

    Args:
        grad_output: Gradient tensor from the next layer, shape (N, C, H_out, W_out)
        output_size: Tuple of (H_out, W_out) - the output size
        input_size: Tuple of (N, C, H_in, W_in) - the original input size
        align_corners: Whether to align corners
        scales_h: Optional scale factor for height
        scales_w: Optional scale factor for width

    Returns:
        Gradient tensor for the input, shape (N, C, H_in, W_in)
    """
    logger.debug("GEMS UPSAMPLE_BICUBIC2D_BACKWARD")

    if grad_output.dim() != 4:
        raise ValueError("grad_output must be a 4D tensor (N, C, H, W)")
    if len(output_size) != 2:
        raise ValueError("output_size must be a tuple of (H_out, W_out)")
    if len(input_size) != 4:
        raise ValueError("input_size must be a tuple of (N, C, H_in, W_in)")

    N, C, H_out, W_out = grad_output.shape
    input_N, input_C, H_in, W_in = input_size

    if N != input_N or C != input_C:
        raise ValueError(
            f"Batch and channel dimensions must match: got N={N}, C={C} "
            f"but input_size expects N={input_N}, C={input_C}"
        )

    device = grad_output.device
    if not grad_output.is_cuda:
        raise ValueError("This Triton kernel requires CUDA tensors")

    # Compute scale factors - backward uses inverse of forward mapping
    if scales_h is not None and scales_w is not None:
        scale_h = scales_h
        scale_w = scales_w
    elif align_corners:
        scale_h = 0.0 if H_out <= 1 else (H_out - 1.0) / (H_in - 1.0)
        scale_w = 0.0 if W_out <= 1 else (W_out - 1.0) / (W_in - 1.0)
    else:
        scale_h = float(H_out) / float(H_in)
        scale_w = float(W_out) / float(W_in)

    # Allocate output gradient tensor
    grad_input = torch.empty((N, C, H_in, W_in), dtype=grad_output.dtype, device=device)

    # Strides
    sN, sC, sH, sW = grad_input.stride()
    oN, oC, oH, oW = grad_output.stride()

    # Grid based on INPUT size
    grid = lambda meta: (triton.cdiv(W_in, meta["BLOCK_W"]) * N * C * H_in,)

    _upsample_bicubic2d_backward_kernel[grid](
        grad_output,
        grad_input,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        sN,
        sC,
        sH,
        sW,
        oN,
        oC,
        oH,
        oW,
        float(scale_h),
        float(scale_w),
        align_corners,
    )

    return grad_input