import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 1, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_D": 1, "BLOCK_H": 8, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 1, "BLOCK_H": 16, "BLOCK_W": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 1, "BLOCK_H": 16, "BLOCK_W": 16}, num_stages=2, num_warps=8),
    ],
    key=["out_d", "out_h", "out_w", "in_d", "in_h", "in_w"],
)
@triton.jit
def adaptive_avg_pool3d_kernel(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Output tensor strides
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    # Input/Output shapes
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Tiling meta-parameters
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (in_n * in_c, out_d * out_h * out_w)
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Calculate n and c indices
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    # Calculate d, h, w indices for output
    num_hw = out_h * out_w
    d_idx = pid_dhw // num_hw
    remaining = pid_dhw % num_hw
    h_idx = remaining // out_w
    w_idx = remaining % out_w

    # Compute the corresponding input region for this output position
    # d_start = floor(d * in_d / out_d), d_end = floor((d+1) * in_d / out_d)
    d_start = d_idx * in_d // out_d
    d_end = (d_idx + 1) * in_d // out_d
    if d_end == d_idx:
        d_end = d_idx + 1
    if d_end > in_d:
        d_end = in_d

    h_start = h_idx * in_h // out_h
    h_end = (h_idx + 1) * in_h // out_h
    if h_end == h_start:
        h_end = h_start + 1
    if h_end > in_h:
        h_end = in_h

    w_start = w_idx * in_w // out_w
    w_end = (w_idx + 1) * in_w // out_w
    if w_end == w_start:
        w_end = w_start + 1
    if w_end > in_w:
        w_end = in_w

    # Compute the sum over the input region
    input_base = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    sum_val = 0.0
    count = 0

    # Iterate through the input region
    for di in range(d_start, d_end):
        for hi in range(h_start, h_end):
            for wi in range(w_start, w_end):
                input_offset = di * in_stride_d + hi * in_stride_h + wi * in_stride_w
                val = tl.load(input_base + input_offset)
                sum_val = sum_val + val
                count = count + 1

    # Compute average
    if count > 0:
        output_val = sum_val / count
    else:
        output_val = 0.0

    # Store the result
    output_base = output_ptr + n_idx * out_stride_n + c_idx * out_stride_c
    output_offset = d_idx * out_stride_d + h_idx * out_stride_h + w_idx * out_stride_w
    tl.store(output_base + output_offset, output_val)


def adaptive_avg_pool3d(input: torch.Tensor, output_size):
    logger.debug("GEMS ADAPTIVE_AVG_POOL3D")

    if input.dim() != 5:
        raise ValueError(f"Expected 5D input, got {input.dim()}D")

    in_n, in_c, in_d, in_h, in_w = input.shape

    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)
    elif len(output_size) == 1:
        output_size = (output_size[0], output_size[0], output_size[0])
    elif len(output_size) != 3:
        raise ValueError(f"Expected output_size to have 1 or 3 elements, got {len(output_size)}")

    out_d, out_h, out_w = output_size

    output = torch.empty((in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype)

    if output.numel() == 0:
        return output

    input = input.contiguous()

    grid = lambda meta: (
        in_n * in_c,
        out_d * out_h * out_w,
    )

    adaptive_avg_pool3d_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.stride(4),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output.stride(4),
        in_n,
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
    )

    return output