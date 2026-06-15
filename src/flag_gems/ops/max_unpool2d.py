import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["out_h", "out_w", "pooled_h", "pooled_w"],
)
@triton.jit
def max_unpool2d_kernel(
    pooled_ptr,
    indices_ptr,
    output_ptr,
    # Input tensor strides (pooled)
    pooled_stride_n,
    pooled_stride_c,
    pooled_stride_h,
    pooled_stride_w,
    # Output tensor strides
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # Shapes
    n,
    c,
    pooled_h,
    pooled_w,
    out_h,
    out_w,
    # Tiling parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(pooled_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // c
    c_idx = pid_nc % c

    h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    pooled_base_ptr = (
        pooled_ptr + n_idx * pooled_stride_n + c_idx * pooled_stride_c
    )
    indices_base_ptr = indices_ptr + n_idx * pooled_stride_n + c_idx * pooled_stride_c

    output_base_ptr = output_ptr + n_idx * out_stride_n + c_idx * out_stride_c

    # Load pooled values and indices
    pooled_offsets = h_offsets[:, None] * pooled_stride_h + w_offsets[None, :] * pooled_stride_w
    indices_offsets = h_offsets[:, None] * pooled_stride_h + w_offsets[None, :] * pooled_stride_w

    pooled_mask = (h_offsets[:, None] < pooled_h) & (w_offsets[None, :] < pooled_w)
    indices_mask = pooled_mask

    pooled_vals = tl.load(pooled_base_ptr + pooled_offsets, mask=pooled_mask, other=0.0)
    indices_flat = tl.load(indices_base_ptr + indices_offsets, mask=indices_mask, other=0)

    # Convert flat indices to 2D positions in output
    h_orig = indices_flat // out_w
    w_orig = indices_flat % out_w

    # Compute output offsets
    out_offsets = h_orig * out_stride_h + w_orig * out_stride_w

    # Store values at original positions
    out_mask = (h_offsets[:, None] < pooled_h) & (w_offsets[None, :] < pooled_w)
    tl.store(output_base_ptr + out_offsets, pooled_vals, mask=out_mask)


def max_unpool2d(pooled: torch.Tensor, indices: torch.Tensor, output_size: list):
    logger.debug("GEMS MAX_UNPOOL2D")

    pooled = pooled.contiguous()
    indices = indices.contiguous()

    n, c, pooled_h, pooled_w = pooled.shape
    out_h, out_w = output_size[0], output_size[1]

    output = torch.zeros(
        (n, c, out_h, out_w), device=pooled.device, dtype=pooled.dtype
    )

    if output.numel() == 0:
        return output

    grid = lambda meta: (
        n * c,
        triton.cdiv(pooled_h, meta["BLOCK_H"]) * triton.cdiv(pooled_w, meta["BLOCK_W"]),
    )

    max_unpool2d_kernel[grid](
        pooled,
        indices,
        output,
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        pooled.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        n,
        c,
        pooled_h,
        pooled_w,
        out_h,
        out_w,
    )

    return output