import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["out_h", "out_w", "in_h", "in_w"],
)
@triton.jit
def adaptive_max_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
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
    # Pooling parameters (computed from output_size)
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    # Meta-parameters for tiling
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

    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h + kh
            w_in = w_out_offsets[None, :] * stride_w + kw
            in_mask = (h_in < in_h) & (w_in < in_w)
            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=min_val
            )
            current_idx = h_in * in_w + w_in

            is_new_max = current_val > max_val_acc
            max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
            max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 16}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 8}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 8, "BLOCK_IN_W": 32}, num_warps=4),
        triton.Config({"BLOCK_IN_H": 32, "BLOCK_IN_W": 32}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 16, "BLOCK_IN_W": 64}, num_warps=8),
        triton.Config({"BLOCK_IN_H": 64, "BLOCK_IN_W": 16}, num_warps=8),
    ],
    key=["in_h", "in_w", "out_h", "out_w"],
)
@triton.jit
def adaptive_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_h,
    in_w,
    out_h,
    out_w,
    # Strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # Pooling parameters
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    # Tiling parameters
    BLOCK_IN_H: tl.constexpr,
    BLOCK_IN_W: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_IN_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks

    h_in_offsets = h_block_idx * BLOCK_IN_H + tl.arange(0, BLOCK_IN_H)
    w_in_offsets = w_block_idx * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)

    current_input_flat_idx = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    grad_acc = tl.zeros((BLOCK_IN_H, BLOCK_IN_W), dtype=tl.float32)

    indices_base_ptr = indices_ptr + nc_idx * out_h * out_w
    grad_output_base_ptr = grad_output_ptr + nc_idx * out_h * out_w

    for kh in tl.static_range(0, kernel_h):
        for kw in tl.static_range(0, kernel_w):
            # h_in_offsets and w_in_offsets are 1D, need to broadcast to 2D for the full mask
            h_out = (h_in_offsets[:, None] - kh) // stride_h
            w_out = (w_in_offsets[None, :] - kw) // stride_w

            # Check if the current input position contributed to this output position
            h_valid = (h_in_offsets[:, None] - kh) % stride_h == 0
            w_valid = (w_in_offsets[None, :] - kw) % stride_w == 0
            out_bounds = (h_out >= 0) & (h_out < out_h) & (w_out >= 0) & (w_out < out_w)
            load_mask = h_valid & w_valid & out_bounds

            safe_h_out = tl.where(load_mask, h_out, 0)
            safe_w_out = tl.where(load_mask, w_out, 0)
            out_offsets = safe_h_out * out_stride_h + safe_w_out

            indices_block = tl.load(
                indices_base_ptr + out_offsets, mask=load_mask, other=-1
            )
            match_mask = indices_block == current_input_flat_idx

            grad_block = tl.load(
                grad_output_base_ptr + out_offsets, mask=match_mask, other=0.0
            )
            grad_acc += grad_block

    grad_input_base_ptr = (
        grad_input_ptr + nc_idx * in_h * in_w
    )
    grad_input_offsets = h_in_offsets[:, None] * in_w + w_in_offsets[None, :]
    store_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask)


def adaptive_max_pool2d(input: torch.Tensor, output_size):
    logger.debug("GEMS ADAPTIVE_MAX_POOL2D FORWARD")
    input = input.contiguous()

    in_n, in_c, in_h, in_w = input.shape

    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    # Compute stride and kernel for adaptive pooling
    stride_h = in_h // out_h
    stride_w = in_w // out_w
    kernel_h = in_h - (out_h - 1) * stride_h
    kernel_w = in_w - (out_w - 1) * stride_w

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    adaptive_max_pool2d_forward_kernel[grid](
        input,
        output,
        indices,
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

    return output, indices


def adaptive_max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
    output_size,
):
    logger.debug("GEMS ADAPTIVE_MAX_POOL2D BACKWARD")
    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    if isinstance(output_size, int):
        out_h_size = out_w_size = output_size
    else:
        out_h_size, out_w_size = output_size

    # Compute stride and kernel for adaptive pooling
    stride_h = in_h // out_h_size
    stride_w = in_w // out_w_size
    kernel_h = in_h - (out_h_size - 1) * stride_h
    kernel_w = in_w - (out_w_size - 1) * stride_w

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_h, meta["BLOCK_IN_H"]) * triton.cdiv(in_w, meta["BLOCK_IN_W"]),
    )

    adaptive_max_pool2d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_h,
        in_w,
        out_h,
        out_w,
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        stride_h,
        stride_w,
        kernel_h,
        kernel_w,
    )

    return grad_input.to(grad_output.dtype)