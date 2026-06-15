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
    restore_value=["running_mean_pointer", "running_var_pointer"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def _native_batch_norm_legit_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    var_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
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
    is_train: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    if is_train:
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
        mean = final_mean

        n = batch_dim * spatial_dim
        # Apply Bessel's correction for the returned variance
        var_bessel = var * n / (n - 1)

        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + var_pointer, var_bessel)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
        )

        inv_std = rsqrt(var + eps)

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

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(
                    curr_input_pointer, mask=mask
                ).to(tl.float32)
                output = weight * (curr_input - mean) * inv_std + bias

                tl.store(
                    curr_output_pointer,
                    output,
                    mask=mask,
                )

    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        var = tl.load(feat_pid + running_var_pointer)
        inv_std = rsqrt(var + eps)

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

                mask = batch_mask[:, None] & spatial_mask[None, :]
                curr_input = tl.load(
                    curr_input_pointer, mask=mask
                ).to(tl.float32)
                output = weight * (curr_input - mean) * inv_std + bias

                tl.store(
                    curr_output_pointer,
                    output,
                    mask=mask,
                )


def _native_batch_norm_legit(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS _NATIVE_BATCH_NORM_LEGIT")

    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    var = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    # Launches 1D grid where each program operates over one feature.
    with torch_device_fn.device(input.device):
        _native_batch_norm_legit_forward_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            mean,
            var,
            output,
            running_mean,
            running_var,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            momentum,
            eps,
            is_train=training,
        )

    if training:
        return output.view_as(input), mean, var
    else:
        # In inference mode, return empty tensors for mean and var
        empty_mean = torch.empty(0, device=input.device, dtype=input.dtype)
        empty_var = torch.empty(0, device=input.device, dtype=input.dtype)
        return output.view_as(input), empty_mean, empty_var