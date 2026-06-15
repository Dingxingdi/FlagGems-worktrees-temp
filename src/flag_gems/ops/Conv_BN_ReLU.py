import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def conv_bn_relu_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        dilation: Dilation.

    Returns:
        Output size of convolution.
    """
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def conv_bn_relu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=1,
    padding=0,
    dilation=1,
):
    """
    Convolution + ReLU fusion.

    This operator performs a 2D convolution followed by ReLU activation.
    It uses the existing conv2d implementation and applies ReLU activation.

    Args:
        input: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, kH, kW)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for convolution
        padding: Padding for convolution
        dilation: Dilation for convolution

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    logger.debug("GEMS CONV_BN_RELU")

    # Import here to avoid circular imports
    from flag_gems.ops.conv2d import conv2d
    from flag_gems.ops.relu import relu

    # First perform convolution
    output = conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)

    # Then apply ReLU activation
    output = relu(output)

    return output