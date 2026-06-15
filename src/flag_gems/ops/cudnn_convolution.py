import logging

import triton  # noqa: F401 - Used by underlying conv2d implementation
from flag_gems.ops.conv2d import conv2d

logger = logging.getLogger(__name__)


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """
    Applies a 2D convolution using cuDNN library.

    This function is a wrapper around conv2d that accepts cuDNN-specific parameters.
    The benchmark, deterministic, and allow_tf32 parameters are passed through to
    the underlying conv2d implementation.

    Args:
        input: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, kH, kW)
        padding: Padding for spatial dimensions
        stride: Stride for spatial dimensions
        dilation: Dilation for spatial dimensions
        groups: Number of blocked groups
        benchmark: Whether to use cuDNN benchmarking (currently ignored by FlagGems)
        deterministic: Whether to use deterministic algorithms (currently ignored by FlagGems)
        allow_tf32: Whether to allow TF32 computation (currently ignored by FlagGems)

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    logger.debug("GEMS CUDNN_CONVOLUTION")
    return conv2d(input, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)