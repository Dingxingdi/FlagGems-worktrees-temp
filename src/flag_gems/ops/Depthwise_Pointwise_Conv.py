import logging

import torch

from flag_gems.ops.conv2d import conv2d, conv2d_output_size

logger = logging.getLogger(__name__)


def depthwise_pointwise_conv2d(
    input,
    depthwise_weight,
    pointwise_weight,
    depthwise_bias=None,
    pointwise_bias=None,
    depthwise_stride=1,
    depthwise_padding=0,
    depthwise_dilation=1,
    pointwise_stride=1,
    pointwise_padding=0,
    pointwise_dilation=1,
):
    """Depthwise Separable Convolution operator.

    This operator performs a depthwise convolution followed by a pointwise convolution,
    which is commonly used in MobileNet and other efficient CNN architectures.

    Args:
        input: Input tensor of shape (N, C_in, H, W)
        depthwise_weight: Depthwise convolution weight of shape (C_in, 1, KH, KW)
        pointwise_weight: Pointwise convolution weight of shape (C_out, C_in, 1, 1)
        depthwise_bias: Optional bias for depthwise convolution
        pointwise_bias: Optional bias for pointwise convolution
        depthwise_stride: Stride for depthwise convolution
        depthwise_padding: Padding for depthwise convolution
        depthwise_dilation: Dilation for depthwise convolution
        pointwise_stride: Stride for pointwise convolution
        pointwise_padding: Padding for pointwise convolution
        pointwise_dilation: Dilation for pointwise convolution

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    logger.debug("GEMS DEPTHWISE_POINTWISE_CONV2D")

    # Validate inputs
    assert input.ndim == 4, f"Input must be 4D, got {input.ndim}D"
    assert depthwise_weight.ndim == 4, f"Depthwise weight must be 4D, got {depthwise_weight.ndim}D"
    assert pointwise_weight.ndim == 4, f"Pointwise weight must be 4D, got {pointwise_weight.ndim}D"

    in_n, in_c, in_h, in_w = input.shape
    _, depthwise_c_per_group, depthwise_kh, depthwise_kw = depthwise_weight.shape

    # Depthwise: groups = in_c, each channel gets its own filter
    assert depthwise_c_per_group == 1, "Depthwise weight must have 1 input channel per group"

    # First perform depthwise convolution
    # Output of depthwise: (N, C_in, H_dw, W_dw)
    depthwise_out = conv2d(
        input,
        depthwise_weight,
        depthwise_bias,
        depthwise_stride,
        depthwise_padding,
        depthwise_dilation,
        groups=in_c,  # Depthwise: each input channel is its own group
    )

    # Calculate output size after depthwise
    if isinstance(depthwise_stride, (list, tuple)):
        stride_h, stride_w = depthwise_stride
    else:
        stride_h = stride_w = depthwise_stride

    if isinstance(depthwise_padding, (list, tuple)):
        padding_h, padding_w = depthwise_padding
    else:
        padding_h = padding_w = depthwise_padding

    if isinstance(depthwise_dilation, (list, tuple)):
        dilation_h, dilation_w = depthwise_dilation
    else:
        dilation_h = dilation_w = depthwise_dilation

    depthwise_out_h = conv2d_output_size(in_h, depthwise_kh, stride_h, padding_h, dilation_h)
    depthwise_out_w = conv2d_output_size(in_w, depthwise_kw, stride_w, padding_w, dilation_w)

    # Pointwise convolution: 1x1 convolution
    # Output of pointwise: (N, C_out, H_out, W_out)
    pointwise_out = conv2d(
        depthwise_out,
        pointwise_weight,
        pointwise_bias,
        pointwise_stride,
        pointwise_padding,
        pointwise_dilation,
        groups=1,  # Pointwise: no depthwise, standard 1x1 conv
    )

    return pointwise_out


# Also export the function with underscore prefix for consistency with other ops
_depthwise_pointwise_conv2d = depthwise_pointwise_conv2d