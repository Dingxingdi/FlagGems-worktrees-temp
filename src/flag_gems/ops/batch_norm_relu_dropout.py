import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.ops.relu import relu
from flag_gems.ops.dropout import dropout
from flag_gems.ops.batch_norm import make_3d_for_bn, batch_norm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


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
    train=True,
):
    """
    Fused BatchNorm + ReLU + Dropout operation.

    Args:
        input: Input tensor
        weight: BatchNorm weight (gamma)
        bias: BatchNorm bias (beta)
        running_mean: Running mean for inference
        running_var: Running variance for inference
        training: Whether to use training mode for BatchNorm
        momentum: BatchNorm momentum
        eps: BatchNorm epsilon
        p: Dropout probability
        train: Whether to apply dropout (training mode)

    Returns:
        Output tensor after BatchNorm -> ReLU -> Dropout
    """
    logger.debug("GEMS BATCH_NORM_RELU_DROPOUT FORWARD")

    # Step 1: Apply BatchNorm
    bn_output, mean, inv_std = batch_norm(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )

    # Step 2: Apply ReLU
    relu_output = relu(bn_output)

    # Step 3: Apply Dropout (only in training mode)
    if train and p > 0 and p < 1:
        dropout_output, mask = dropout(relu_output, p=p, train=train)
        return dropout_output, mean, inv_std, mask
    elif p == 0 or not train:
        # No dropout when p=0 or not training
        return relu_output, mean, inv_std, None
    else:
        # p == 1: all zeros
        return torch.zeros_like(relu_output), mean, inv_std, None


def batch_norm_relu_dropout_backward(
    grad_output,
    input,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_invstd=None,
    train=False,
    eps=1e-05,
    p=0.5,
    mask=None,
    output_mask=None,
):
    """
    Backward pass for fused BatchNorm + ReLU + Dropout.

    Args:
        grad_output: Gradient of output
        input: Original input
        weight: BatchNorm weight
        running_mean: Running mean
        running_var: Running variance
        save_mean: Saved mean from forward pass
        save_invstd: Saved inv_std from forward pass
        train: Training mode flag
        eps: BatchNorm epsilon
        p: Dropout probability
        mask: Dropout mask from forward pass
        output_mask: Which gradients to return

    Returns:
        Gradients for input, weight, bias
    """
    logger.debug("GEMS BATCH_NORM_RELU_DROPOUT BACKWARD")

    # Step 1: Dropout backward
    if mask is not None and p > 0 and p < 1:
        scale = 1.0 / (1.0 - p)
        from flag_gems.ops.dropout import dropout_backward

        dropout_grad = dropout_backward(grad_output, mask, scale)
    else:
        dropout_grad = grad_output

    # Step 2: ReLU backward
    # We need to recompute the ReLU input (bn_output) for the mask
    # Since we don't have it saved, we need to reconstruct from input
    input_3d = make_3d_for_bn(input)
    batch_dim, feat_dim, spatial_dim = input_3d.shape

    # Compute BatchNorm forward to get the output for ReLU mask
    mean = save_mean
    inv_std = save_invstd

    bn_output_3d = torch.empty_like(input_3d)
    for feat_idx in range(feat_dim):
        if weight is not None:
            w = weight[feat_idx] if weight.dim() > 0 else weight
        else:
            w = 1.0
        if bias is not None:
            b = bias[feat_idx] if bias.dim() > 0 else bias
        else:
            b = 0.0

        # Get the mean and inv_std for this feature
        mean_val = mean[feat_idx] if mean is not None else 0.0
        inv_std_val = inv_std[feat_idx] if inv_std is not None else 1.0

        # Compute normalized output
        input_slice = input_3d[:, feat_idx, :]
        normalized = (input_slice - mean_val) * inv_std_val
        bn_slice = w * normalized + b
        bn_output_3d[:, feat_idx, :] = bn_slice

    bn_output = bn_output_3d.view_as(input)

    # Apply ReLU backward (zero out gradients where input <= 0)
    relu_mask = bn_output > 0
    relu_grad = dropout_grad * relu_mask

    # Step 3: BatchNorm backward
    from flag_gems.ops.batch_norm import batch_norm_backward

    bn_grad = batch_norm_backward(
        relu_grad,
        input,
        weight=weight,
        running_mean=running_mean,
        running_var=running_var,
        save_mean=save_mean,
        save_invstd=save_invstd,
        train=train,
        eps=eps,
        output_mask=output_mask,
    )

    return bn_grad