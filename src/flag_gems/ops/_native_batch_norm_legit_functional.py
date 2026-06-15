import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.ops.batch_norm import make_3d_for_bn

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
    restore_value=["running_mean_pointer", "running_var_pointer"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def _native_batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
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

    # Training mode: compute batch statistics and update running stats
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
        inv_std = rsqrt(var + eps)
        mean = final_mean

        # Store mean and inv_std in float32
        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + inv_std_pointer, inv_std)
        # Store variance in float32
        tl.store(feat_pid + var_pointer, var)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        n = batch_dim * spatial_dim
        new_running_mean = (1 - momentum) * running_mean + momentum * mean
        new_running_var = (1 - momentum) * running_var + momentum * var * n / (n - 1)
        tl.store(running_mean_pointer, new_running_mean)
        tl.store(running_var_pointer, new_running_var)

    else:
        # Inference mode: use running statistics
        mean = tl.load(feat_pid + running_mean_pointer)
        var = tl.load(feat_pid + running_var_pointer)
        inv_std = rsqrt(var + eps)
        # Store for consistency
        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + inv_std_pointer, inv_std)
        tl.store(feat_pid + var_pointer, var)

    # Load weight and bias
    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0

    # Normalize and scale
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


def _native_batch_norm_legit_functional(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("GEMS _NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")

    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    # Always use float32 for save_mean, save_var to match PyTorch behavior
    mean = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    var = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    # Launches 1D grid where each program operates over one feature.
    with torch_device_fn.device(input.device):
        _native_batch_norm_forward_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
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

    # inv_std is always computed in float32 - use it directly as save_var
    # For both training and inference, mean contains the correct value (batch mean for
    # training, running_mean for inference, both stored in float32 by the kernel)
    save_mean = mean
    save_var = inv_std

    return output.view_as(input), save_mean, save_var, running_mean, running_var