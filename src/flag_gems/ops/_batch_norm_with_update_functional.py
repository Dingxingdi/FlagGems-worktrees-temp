import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
    """
    Converts the input to a 3D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)

    elif input.ndim >= 4:
        input = input.flatten(2, -1)

    return input


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def _batch_norm_with_update_functional_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    running_mean_out_pointer,
    running_var_out_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    # Compute mean and variance over batch and spatial dims
    mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
    n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)

    for m_step in range(0, m_num_steps):
        for n_step in range(0, n_num_steps):
            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_pointer, mask=mask).to(tl.float32)

            step = m_step * n_num_steps + n_step + 1
            new_mean = tl.where(mask, mean + (curr_input - mean) / step, mean)
            new_var = tl.where(
                mask, var + (curr_input - new_mean) * (curr_input - mean), var
            )
            cnt += mask.to(tl.int32)
            mean = new_mean
            var = new_var

    final_mean = tl.sum(mean * cnt) / (batch_dim * spatial_dim)
    var = tl.sum(var + cnt * (mean - final_mean) * (mean - final_mean)) / (
        batch_dim * spatial_dim
    )
    inv_std = rsqrt(var + eps)
    mean = final_mean

    tl.store(feat_pid + mean_pointer, mean)
    tl.store(feat_pid + inv_std_pointer, inv_std)

    running_mean_pointer += feat_pid
    running_var_pointer += feat_pid
    running_mean_out_pointer += feat_pid
    running_var_out_pointer += feat_pid

    running_mean = tl.load(running_mean_pointer)
    running_var = tl.load(running_var_pointer)

    n = batch_dim * spatial_dim
    # Compute updated running mean and var (not in-place)
    new_running_mean = (1 - momentum) * running_mean + momentum * mean
    new_running_var = (1 - momentum) * running_var + momentum * var * n / (n - 1)

    # Store to output tensors (not in-place)
    tl.store(running_mean_out_pointer, new_running_mean)
    tl.store(running_var_out_pointer, new_running_var)

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_pointer = (
                output_pointer
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            output = weight * (curr_input - mean) * inv_std + bias

            tl.store(
                curr_output_pointer,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


def _batch_norm_with_update_functional(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS _BATCH_NORM_WITH_UPDATE_FUNCTIONAL")

    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

    # Create output tensors for updated running stats
    running_mean_out = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    running_var_out = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

    # Launches 1D grid where each program operates over one feature.
    with torch_device_fn.device(input.device):
        _batch_norm_with_update_functional_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
            output,
            running_mean,
            running_var,
            running_mean_out,
            running_var_out,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            momentum,
            eps,
        )

    # Output: (output, mean, invstd, reserve_empty, running_mean_out, running_var_out)
    reserve = torch.empty(0, dtype=torch.uint8, device=input.device)
    return (
        output.view_as(input),
        mean,
        inv_std,
        reserve,
        running_mean_out,
        running_var_out,
    )