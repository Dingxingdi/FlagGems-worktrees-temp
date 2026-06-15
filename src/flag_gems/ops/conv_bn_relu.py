import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Helper function for conv2d output size calculation
def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# BN parameters: running_mean, running_var, weight, bias, eps
def fused_conv_bn_relu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    eps=1e-5,
):
    """
    Fused Conv2d + BatchNorm + ReLU operator.

    Args:
        input: Input tensor of shape (N, C_in, H, W)
        weight: Conv weight tensor of shape (C_out, C_in//groups, kH, kW)
        bias: Conv bias tensor of shape (C_out,), can be None
        running_mean: BatchNorm running mean of shape (C_out,)
        running_var: BatchNorm running variance of shape (C_out,)
        bn_weight: BatchNorm weight (gamma) of shape (C_out,)
        bn_bias: BatchNorm bias (beta) of shape (C_out,)
        stride: Conv stride
        padding: Conv padding
        dilation: Conv dilation
        groups: Conv groups
        eps: BatchNorm epsilon

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    logger.debug(
        "GEMS FUSED_CONV_BN_RELU FORWARD, [input shape]: %s, [weight shape]: %s",
        input.size(),
        weight.size(),
    )

    # First perform conv2d
    # We'll call the existing conv2d implementation
    from flag_gems.ops.conv2d import conv2d as gems_conv2d

    # Compute conv output shape
    in_n, in_c, in_h, in_w = input.shape
    out_c, weight_c, k_h, k_w = weight.shape

    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (list, tuple)):
        padding_h, padding_w = padding
    else:
        padding_h = padding_w = padding

    if isinstance(dilation, (list, tuple)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    out_h = conv2d_output_size(in_h, k_h, stride_h, padding_h, dilation_h)
    out_w = conv2d_output_size(in_w, k_w, stride_w, padding_w, dilation_w)

    # Perform conv2d
    conv_out = gems_conv2d(input, weight, bias, stride, padding, dilation, groups)

    # Then perform batch norm
    # Use the existing batch_norm implementation
    from flag_gems.ops.batch_norm import batch_norm as gems_batch_norm

    # Batch norm expects input in (N, C, H, W) format which conv_out already is
    # running_mean and running_var should be on the same device
    # batch_norm returns (output, mean, inv_std), we only need output
    bn_out, _, _ = gems_batch_norm(
        conv_out,
        bn_weight,
        bn_bias,
        running_mean,
        running_var,
        training=False,  # Inference mode
        momentum=0.0,
        eps=eps,
    )

    # Finally perform relu
    # Use the existing relu implementation
    from flag_gems.ops.relu import relu as gems_relu

    output = gems_relu(bn_out)

    return output


def conv_bn_relu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    eps=1e-5,
):
    """
    Fused Conv2d + BatchNorm + ReLU operator.
    Alias for fused_conv_bn_relu.
    """
    return fused_conv_bn_relu(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        stride,
        padding,
        dilation,
        groups,
        eps,
    )


# Conv+BN+ReLU version with + in the name
def Conv_BN_ReLU(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    eps=1e-5,
):
    """
    Fused Conv2d + BatchNorm + ReLU operator.
    """
    return fused_conv_bn_relu(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        stride,
        padding,
        dilation,
        groups,
        eps,
    )