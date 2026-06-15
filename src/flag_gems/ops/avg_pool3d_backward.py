import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 8, "BLOCK_W": 16}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_D": 4, "BLOCK_H": 16, "BLOCK_W": 8}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_D": 8, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=3, num_warps=4),
    ],
    key=["in_d", "in_h", "in_w", "kernel_d", "kernel_h", "kernel_w", "stride_d", "stride_h", "stride_w"],
)
@triton.jit
def avg_pool3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    # Input/Output shapes
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    # AvgPool specific parameters
    COUNT_INCLUDE_PAD: tl.constexpr,
    divisor_override,
    # Tiling meta-parameters
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(in_h, BLOCK_H)
    num_w_blocks = tl.cdiv(in_w, BLOCK_W)

    d_block_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    h_block_idx = (pid_dhw // num_w_blocks) % num_h_blocks
    w_block_idx = pid_dhw % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    grad_input_base_ptr = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c
    grad_output_base_ptr = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_c

    d_in_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    grad_acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

    for kd_loop in range(kernel_d):
        for kh_loop in range(kernel_h):
            for kw_loop in range(kernel_w):
                d_out_num = d_in_offsets[:, None, None] + padding_d - kd_loop
                h_out_num = h_in_offsets[None, :, None] + padding_h - kh_loop
                w_out_num = w_in_offsets[None, None, :] + padding_w - kw_loop

                d_valid_map = (d_out_num >= 0) & ((d_out_num % stride_d) == 0)
                h_valid_map = (h_out_num >= 0) & ((h_out_num % stride_h) == 0)
                w_valid_map = (w_out_num >= 0) & ((w_out_num % stride_w) == 0)

                d_out = d_out_num // stride_d
                h_out = h_out_num // stride_h
                w_out = w_out_num // stride_w

                d_out_mask = d_valid_map & (d_out < out_d)
                h_out_mask = h_valid_map & (h_out < out_h)
                w_out_mask = w_valid_map & (w_out < out_w)
                out_mask = d_out_mask & h_out_mask & w_out_mask

                if divisor_override != 0:
                    divisor = tl.full(
                        (BLOCK_D, BLOCK_H, BLOCK_W), divisor_override, dtype=tl.float32
                    )
                elif COUNT_INCLUDE_PAD:
                    divisor = tl.full(
                        (BLOCK_D, BLOCK_H, BLOCK_W),
                        kernel_d * kernel_h * kernel_w,
                        dtype=tl.float32,
                    )
                else:
                    d_start = d_out * stride_d - padding_d
                    h_start = h_out * stride_h - padding_h
                    w_start = w_out * stride_w - padding_w

                    count = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.int32)
                    for kd_count in range(0, kernel_d):
                        for kh_count in range(0, kernel_h):
                            for kw_count in range(0, kernel_w):
                                d_in_for_count = d_start + kd_count
                                h_in_for_count = h_start + kh_count
                                w_in_for_count = w_start + kw_count
                                is_valid = (
                                    (d_in_for_count >= 0)
                                    & (d_in_for_count < in_d)
                                    & (h_in_for_count >= 0)
                                    & (h_in_for_count < in_h)
                                    & (w_in_for_count >= 0)
                                    & (w_in_for_count < in_w)
                                )
                                count += is_valid.to(tl.int32)
                    divisor = count.to(tl.float32)

                divisor = tl.where(divisor == 0, 1.0, divisor)

                # Compute linear offset for output
                out_offset = d_out * out_stride_d + h_out * out_stride_h + w_out * out_stride_w
                grad_out_ptr = grad_output_base_ptr + out_offset
                grad_out_val = tl.load(grad_out_ptr, mask=out_mask, other=0.0)
                grad_acc += tl.where(out_mask, grad_out_val / divisor, 0.0)

    grad_input_store_ptr = (
        grad_input_base_ptr
        + d_in_offsets[:, None, None] * in_stride_d
        + h_in_offsets[None, :, None] * in_stride_h
        + w_in_offsets[None, None, :] * in_stride_w
    )
    in_write_mask = (
        (d_in_offsets[:, None, None] < in_d)
        & (h_in_offsets[None, :, None] < in_h)
        & (w_in_offsets[None, None, :] < in_w)
    )
    tl.store(
        grad_input_store_ptr,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=in_write_mask,
    )


def _parse_pool3d_params(kernel_size, stride, padding):
    if isinstance(kernel_size, int):
        kernel_d = kernel_h = kernel_w = kernel_size
    else:
        kernel_d, kernel_h, kernel_w = kernel_size

    if stride is None or (isinstance(stride, (list, tuple)) and not stride):
        stride_d = stride_h = stride_w = kernel_d
    elif isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(padding, int):
        padding_d = padding_h = padding_w = padding
    else:
        padding_d, padding_h, padding_w = padding

    if stride_d <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("stride must be greater than zero")

    if padding_d < 0 or padding_h < 0 or padding_w < 0:
        raise ValueError("padding must be non-negative")

    return kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w


def avg_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    logger.debug("GEMS AVG_POOL3D_BACKWARD")

    if divisor_override is not None and divisor_override == 0:
        raise ValueError("divisor_override cannot be zero")

    grad_output = grad_output.contiguous()

    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
    ) = _parse_pool3d_params(kernel_size, stride, padding)

    in_n, in_c, in_d, in_h, in_w = input.shape
    out_d, out_h, out_w = grad_output.shape[2], grad_output.shape[3], grad_output.shape[4]

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_output.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_d, meta["BLOCK_D"])
        * triton.cdiv(in_h, meta["BLOCK_H"])
        * triton.cdiv(in_w, meta["BLOCK_W"]),
    )

    avg_pool3d_backward_kernel[grid](
        grad_output,
        grad_input,
        in_n,
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_input.stride(4),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_output.stride(4),
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        COUNT_INCLUDE_PAD=count_include_pad,
        divisor_override=divisor_override if divisor_override is not None else 0.0,
    )

    return grad_input.to(grad_output.dtype)