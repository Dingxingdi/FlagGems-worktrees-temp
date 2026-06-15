import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

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


# Kernel for fused BatchNorm + ReLU + Dropout forward pass
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
    restore_value=["running_mean_pointer", "running_var_pointer"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def batch_norm_relu_dropout_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    dropout_mask_pointer,
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
    p,
    is_train: tl.constexpr,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    # Training mode: compute mean and variance
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

        tl.store(feat_pid + mean_pointer, mean)
        tl.store(feat_pid + inv_std_pointer, inv_std)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        n = batch_dim * spatial_dim
        tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
        )

    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0

    # Pre-compute dropout condition (Triton doesn't support chained and/or)
    p_gt_zero = p > 0.0
    p_lt_one = p < 1.0
    dropout_enabled = is_train and p_gt_zero
    dropout_training = dropout_enabled and p_lt_one

    # Setup dropout random number generation (always convert, but only use if training)
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    UNROLL = 4
    dropout_scale = 1.0 / (1.0 - p)

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
            curr_input = tl.load(curr_input_pointer, mask=mask).to(tl.float32)

            # BatchNorm: (x - mean) * inv_std * weight + bias
            output = weight * (curr_input - mean) * inv_std + bias

            # ReLU: max(0, output)
            output = tl.where(output > 0, output, 0.0)

            # Dropout (if training and p > 0)
            if dropout_training:
                # Generate random numbers for dropout mask
                base_offset = (m_step * tl.cdiv(spatial_dim, BLOCK_N) + n_step) * BLOCK_M * BLOCK_N
                i4 = base_offset + batch_offset[:, None] * BLOCK_N + spatial_offset[None, :]
                c0_offset = i4 & 0xFFFFFFFF
                c1_offset = (i4 >> 32) & 0xFFFFFFFF
                _O = c0_offset * 0

                r0, r1, r2, r3 = tl.philox(philox_seed, c0_offset, c1_offset, _O, _O)
                r0 = uint_to_uniform_float(r0)
                r1 = uint_to_uniform_float(r1)
                r2 = uint_to_uniform_float(r2)
                r3 = uint_to_uniform_float(r3)

                # Create dropout mask
                mask0 = r0 > p
                mask1 = r1 > p
                mask2 = r2 > p
                mask3 = r3 > p

                # Apply dropout scale
                output = output * dropout_scale * tl.where(mask0, 1.0, 0.0)

            tl.store(
                curr_output_pointer,
                output,
                mask=mask,
            )


def batch_norm_relu_dropout(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
    p=0.5,
):
    """
    Fused BatchNorm + ReLU + Dropout operation.

    Args:
        input: Input tensor
        weight: BatchNorm weight (scale)
        bias: BatchNorm bias
        running_mean: Running mean for inference
        running_var: Running variance for inference
        training: Whether in training mode
        momentum: BatchNorm momentum
        eps: BatchNorm epsilon
        p: Dropout probability

    Returns:
        Output tensor
    """
    logger.debug("GEMS BATCHNORM_RELU_DROPOUT FORWARD")

    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    dropout_mask = None
    if training and p > 0.0 and p < 1.0:
        dropout_mask = torch.empty_like(input_3d, dtype=torch.bool)

    # Generate random seed and offset for dropout
    philox_seed = 0
    philox_offset = 0
    if training and p > 0.0 and p < 1.0:
        philox_seed, philox_offset = philox_backend_seed_offset(input_3d.numel())

    # Launches 1D grid where each program operates over one feature.
    with torch_device_fn.device(input.device):
        batch_norm_relu_dropout_forward_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
            output,
            running_mean,
            running_var,
            dropout_mask,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            momentum,
            eps,
            p,
            is_train=training,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
        )

    return output.view_as(input)