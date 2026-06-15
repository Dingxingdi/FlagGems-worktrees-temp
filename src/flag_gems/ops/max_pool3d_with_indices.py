import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool3d_output_size(
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
        # PyTorch-compatible adjustment for ceil_mode
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


@libentry()
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_T": 4, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_T": 4, "BLOCK_H": 8, "BLOCK_W": 16}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_T": 4, "BLOCK_H": 16, "BLOCK_W": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_T": 2, "BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_T": 2, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_T": 8, "BLOCK_H": 8, "BLOCK_W": 8}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_T": 1, "BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=8
        ),
    ],
    key=[
        "out_t",
        "out_h",
        "out_w",
        "kernel_t",
        "kernel_h",
        "kernel_w",
        "stride_t",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_t,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_t,
    in_h,
    in_w,
    out_t,
    out_h,
    out_w,
    # Pooling parameters
    kernel_t: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_t: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_t: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_t: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_thw = tl.program_id(1)

    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    t_block_idx = pid_thw // (num_h_blocks * num_w_blocks)
    remaining = pid_thw % (num_h_blocks * num_w_blocks)
    h_block_idx = remaining // num_w_blocks
    w_block_idx = remaining % num_w_blocks

    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    t_out_offsets = t_block_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_T, BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_T, BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kt in tl.static_range(0, kernel_t):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                t_in = (
                    t_out_offsets[:, None, None] * stride_t
                    - padding_t
                    + kt * dilation_t
                )
                h_in = (
                    h_out_offsets[None, :, None] * stride_h
                    - padding_h
                    + kh * dilation_h
                )
                w_in = (
                    w_out_offsets[None, None, :] * stride_w
                    - padding_w
                    + kw * dilation_w
                )
                in_mask = (
                    (t_in >= 0)
                    & (t_in < in_t)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )
                input_offset = (
                    t_in * in_stride_t + h_in * in_stride_h + w_in * in_stride_w
                )
                current_val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=min_val
                )
                current_idx = t_in * in_h * in_w + h_in * in_w + w_in

                is_new_max = current_val > max_val_acc
                max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
                max_idx_acc = tl.where(
                    is_new_max & in_mask, current_idx, max_idx_acc
                )

    out_base_ptr = output_ptr + pid_nc * out_t * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_t * out_h * out_w

    output_block_ptr = (
        out_base_ptr
        + t_out_offsets[:, None, None] * out_h * out_w
        + h_out_offsets[None, :, None] * out_w
        + w_out_offsets[None, None, :]
    )
    indices_block_ptr = (
        indices_base_ptr
        + t_out_offsets[:, None, None] * out_h * out_w
        + h_out_offsets[None, :, None] * out_w
        + w_out_offsets[None, None, :]
    )

    out_mask = (
        (t_out_offsets[:, None, None] < out_t)
        & (h_out_offsets[None, :, None] < out_h)
        & (w_out_offsets[None, None, :] < out_w)
    )
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_IN_T": 4, "BLOCK_IN_H": 8, "BLOCK_IN_W": 8}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_IN_T": 4, "BLOCK_IN_H": 8, "BLOCK_IN_W": 16}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_IN_T": 4, "BLOCK_IN_H": 16, "BLOCK_IN_W": 8}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_IN_T": 2, "BLOCK_IN_H": 16, "BLOCK_IN_W": 16}, num_warps=8
        ),
        triton.Config(
            {"BLOCK_IN_T": 8, "BLOCK_IN_H": 8, "BLOCK_IN_W": 8}, num_warps=4
        ),
        triton.Config(
            {"BLOCK_IN_T": 1, "BLOCK_IN_H": 16, "BLOCK_IN_W": 16}, num_warps=8
        ),
    ],
    key=[
        "in_t",
        "in_h",
        "in_w",
        "kernel_t",
        "kernel_h",
        "kernel_w",
        "stride_t",
        "stride_h",
        "stride_w",
    ],
)
@triton.jit
def max_pool3d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_t,
    in_h,
    in_w,
    out_t,
    out_h,
    out_w,
    # Strides for grad_output/indices
    out_stride_nc,
    out_stride_t,
    out_stride_h,
    out_stride_w,
    # Pooling parameters
    kernel_t: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_t: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_t: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_t: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Tiling parameters
    BLOCK_IN_T: tl.constexpr,
    BLOCK_IN_H: tl.constexpr,
    BLOCK_IN_W: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    pid_thw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_IN_W)
    num_h_blocks = tl.cdiv(in_h, BLOCK_IN_H)
    t_block_idx = pid_thw // (num_h_blocks * num_w_blocks)
    remaining = pid_thw % (num_h_blocks * num_w_blocks)
    h_block_idx = remaining // num_w_blocks
    w_block_idx = remaining % num_w_blocks

    t_in_offsets = t_block_idx * BLOCK_IN_T + tl.arange(0, BLOCK_IN_T)
    h_in_offsets = h_block_idx * BLOCK_IN_H + tl.arange(0, BLOCK_IN_H)
    w_in_offsets = w_block_idx * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)

    current_input_flat_idx = (
        t_in_offsets[:, None, None] * in_h * in_w
        + h_in_offsets[None, :, None] * in_w
        + w_in_offsets[None, None, :]
    )
    grad_acc = tl.zeros((BLOCK_IN_T, BLOCK_IN_H, BLOCK_IN_W), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_stride_nc
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_stride_nc

    for kt in tl.static_range(0, kernel_t):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                numerator_t = (
                    t_in_offsets[:, None, None] + padding_t - kt * dilation_t
                )
                numerator_h = (
                    h_in_offsets[None, :, None] + padding_h - kh * dilation_h
                )
                numerator_w = (
                    w_in_offsets[None, None, :] + padding_w - kw * dilation_w
                )

                valid_map_mask = (
                    (numerator_t % stride_t == 0)
                    & (numerator_h % stride_h == 0)
                    & (numerator_w % stride_w == 0)
                )
                t_out = numerator_t // stride_t
                h_out = numerator_h // stride_h
                w_out = numerator_w // stride_w
                out_bounds_mask = (
                    (t_out >= 0)
                    & (t_out < out_t)
                    & (h_out >= 0)
                    & (h_out < out_h)
                    & (w_out >= 0)
                    & (w_out < out_w)
                )
                load_mask = valid_map_mask & out_bounds_mask

                safe_t_out = tl.where(load_mask, t_out, 0)
                safe_h_out = tl.where(load_mask, h_out, 0)
                safe_w_out = tl.where(load_mask, w_out, 0)
                out_offsets = (
                    safe_t_out * out_stride_t
                    + safe_h_out * out_stride_h
                    + safe_w_out
                )

                indices_block = tl.load(
                    indices_base_ptr + out_offsets, mask=load_mask, other=-1
                )
                match_mask = indices_block == current_input_flat_idx

                grad_block = tl.load(
                    grad_output_base_ptr + out_offsets, mask=match_mask, other=0.0
                )
                grad_acc += grad_block

    grad_input_base_ptr = grad_input_ptr + nc_idx * in_t * in_h * in_w
    grad_input_offsets = (
        t_in_offsets[:, None, None] * in_h * in_w
        + h_in_offsets[None, :, None] * in_w
        + w_in_offsets[None, None, :]
    )
    store_mask = (
        (t_in_offsets[:, None, None] < in_t)
        & (h_in_offsets[None, :, None] < in_h)
        & (w_in_offsets[None, None, :] < in_w)
    )
    tl.store(
        grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask
    )


def _parse_pool_params_3d(kernel_size, stride, padding, dilation):
    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return param
        raise ValueError(f"Invalid {name}: {param}")

    kernel_t, kernel_h, kernel_w = _parse_param(kernel_size, "kernel_size")
    stride_t, stride_h, stride_w = _parse_param(
        stride, "stride", default=(kernel_t, kernel_h, kernel_w)
    )
    padding_t, padding_h, padding_w = _parse_param(
        padding, "padding", default=(0, 0, 0)
    )
    dilation_t, dilation_h, dilation_w = _parse_param(
        dilation, "dilation", default=(1, 1, 1)
    )

    if stride_t <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, but got stride=({stride_t}, {stride_h}, {stride_w})"
        )
    if padding_t < 0 or padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({padding_t}, {padding_h}, {padding_w})"
        )
    if dilation_t <= 0 or dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dilation_t}, {dilation_h}, {dilation_w})"
        )

    return (
        kernel_t,
        kernel_h,
        kernel_w,
        stride_t,
        stride_h,
        stride_w,
        padding_t,
        padding_h,
        padding_w,
        dilation_t,
        dilation_h,
        dilation_w,
    )


def max_pool3d_with_indices(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS MAX_POOL3D_WITH_INDICES FORWARD")
    input = input.contiguous()

    params = _parse_pool_params_3d(kernel_size, stride, padding, dilation)
    (
        kernel_t,
        kernel_h,
        kernel_w,
        stride_t,
        stride_h,
        stride_w,
        padding_t,
        padding_h,
        padding_w,
        dilation_t,
        dilation_h,
        dilation_w,
    ) = params

    if input.dim() != 5:
        raise ValueError(
            f"max_pool3d_with_indices expects 5D input (NCDHW), got {input.dim()}D"
        )
    in_n, in_c, in_t, in_h, in_w = input.shape

    out_t = max_pool3d_output_size(
        in_t, kernel_t, stride_t, padding_t, dilation_t, ceil_mode
    )
    out_h = max_pool3d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = max_pool3d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    output = torch.empty(
        (in_n, in_c, out_t, out_h, out_w),
        device=input.device,
        dtype=input.dtype,
    )
    indices = torch.empty(
        (in_n, in_c, out_t, out_h, out_w),
        device=input.device,
        dtype=torch.int64,
    )

    if output.numel() == 0:
        return output, indices

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_t, meta["BLOCK_T"])
        * triton.cdiv(out_h, meta["BLOCK_H"])
        * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    max_pool3d_forward_kernel[grid](
        input,
        output,
        indices,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.stride(4),
        in_c,
        in_t,
        in_h,
        in_w,
        out_t,
        out_h,
        out_w,
        kernel_t,
        kernel_h,
        kernel_w,
        stride_t,
        stride_h,
        stride_w,
        padding_t,
        padding_h,
        padding_w,
        dilation_t,
        dilation_h,
        dilation_w,
    )

    return output, indices


def max_pool3d_with_indices_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    logger.debug("GEMS MAX_POOL3D_WITH_INDICES BACKWARD")
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    params = _parse_pool_params_3d(kernel_size, stride, padding, dilation)
    (
        kernel_t,
        kernel_h,
        kernel_w,
        stride_t,
        stride_h,
        stride_w,
        padding_t,
        padding_h,
        padding_w,
        dilation_t,
        dilation_h,
        dilation_w,
    ) = params

    if input.dim() != 5:
        raise ValueError(
            f"max_pool3d_with_indices_backward expects 5D input (NCDHW), got {input.dim()}D"
        )
    in_n, in_c, in_t, in_h, in_w = input.shape

    out_t, out_h, out_w = grad_output.shape[2], grad_output.shape[3], grad_output.shape[4]

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_t, meta["BLOCK_IN_T"])
        * triton.cdiv(in_h, meta["BLOCK_IN_H"])
        * triton.cdiv(in_w, meta["BLOCK_IN_W"]),
    )

    out_stride_nc = out_t * out_h * out_w
    out_stride_t = out_h * out_w
    out_stride_h = out_w
    out_stride_w = 1

    max_pool3d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_t,
        in_h,
        in_w,
        out_t,
        out_h,
        out_w,
        out_stride_nc,
        out_stride_t,
        out_stride_h,
        out_stride_w,
        kernel_t,
        kernel_h,
        kernel_w,
        stride_t,
        stride_h,
        stride_w,
        padding_t,
        padding_h,
        padding_w,
        dilation_t,
        dilation_h,
        dilation_w,
    )

    return grad_input.to(grad_output.dtype)