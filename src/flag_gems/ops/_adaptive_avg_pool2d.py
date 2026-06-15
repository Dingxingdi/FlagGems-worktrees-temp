import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _adaptive_pooling_output_size(in_size: int, out_size: int) -> tuple:
    """Compute stride and kernel size for adaptive pooling."""
    if out_size == 1:
        return 1, in_size
    stride = in_size // out_size
    kernel_size = in_size - (out_size - 1) * stride
    return stride, kernel_size


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["out_h", "out_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Adaptive pooling parameters (computed from output size)
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    sum_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # For adaptive pooling, kernel_h and kernel_w can vary based on output size
    # We need to iterate over all elements in the pooling window
    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h + kh
            w_in = w_out_offsets[None, :] * stride_w + kw
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)

            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=0.0
            )

            sum_acc += tl.where(in_mask, current_val, 0.0)
            count_acc += in_mask.to(tl.int32)

    # Divide by count (adaptive avg pool always uses actual count)
    divisor = count_acc.to(tl.float32)
    output_vals = tl.where(divisor != 0, sum_acc / divisor, 0.0)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(
        output_block_ptr, output_vals.to(output_ptr.type.element_ty), mask=out_mask
    )


def _adaptive_avg_pool2d(input: torch.Tensor, output_size: tuple):
    logger.debug("GEMS ADAPTIVE_AVG_POOL2D")

    input = input.contiguous()

    in_n, in_c, in_h, in_w = input.shape

    if output_size is None:
        output_size = (1, 1)

    if len(output_size) != 2:
        raise ValueError("output_size must be a tuple of 2 ints")

    out_h, out_w = output_size

    if out_h <= 0 or out_w <= 0:
        raise ValueError("output_size must be positive")

    # Compute stride and kernel size for each dimension
    stride_h, kernel_h = _adaptive_pooling_output_size(in_h, out_h)
    stride_w, kernel_w = _adaptive_pooling_output_size(in_w, out_w)

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )

    if output.numel() == 0:
        return output

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    adaptive_avg_pool2d_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        stride_h,
        stride_w,
        kernel_h,
        kernel_w,
    )

    return output