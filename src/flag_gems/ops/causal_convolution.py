import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def causal_conv1d_forward_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_length_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_kernel_stride,
    output_batch_stride,
    output_channel_stride,
    output_length_stride,
    batch_size,
    in_channels,
    out_channels,
    length,
    kernel_size,
    stride: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Causal 1D convolution kernel.
    Output at position t only depends on input at positions <= t (causal property).
    """
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    # Each program computes one output element
    for idx in range(pid, batch_size * out_channels * length, num_pid):
        # Decode idx to batch, out_channel, length positions
        tmp = idx
        batch_idx = tmp // (out_channels * length)
        tmp = tmp % (out_channels * length)
        out_channel_idx = tmp // length
        length_idx = tmp % length

        # Calculate the start position in the original (unpadded) input
        # For causal convolution, we apply left padding
        input_start = length_idx * stride

        # Compute the convolution
        result = 0.0

        for k in range(kernel_size):
            # Input position for this kernel element
            input_pos = input_start + k * dilation

            # For causal convolution, we need to check against the padded input
            # The padded input has additional elements at the beginning
            if input_pos >= 0:
                # Load input value - sum over in_channels
                for in_ch in range(in_channels):
                    input_offset = (
                        batch_idx * input_batch_stride
                        + in_ch * input_channel_stride
                        + input_pos * input_length_stride
                    )
                    input_val = tl.load(input_ptr + input_offset)

                    # Load weight
                    weight_offset = (
                        out_channel_idx * weight_out_channel_stride
                        + in_ch * weight_in_channel_stride
                        + k * weight_kernel_stride
                    )
                    weight_val = tl.load(weight_ptr + weight_offset)

                    result += input_val * weight_val

        # Store the result
        output_offset = (
            batch_idx * output_batch_stride
            + out_channel_idx * output_channel_stride
            + length_idx * output_length_stride
        )
        tl.store(output_ptr + output_offset, result)


def causal_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Causal 1D convolution.

    Causal convolution ensures that output at position t only depends on input
    at positions <= t. This is achieved by applying padding only to the left
    (beginning) of the input sequence.

    Args:
        input: Input tensor of shape (batch, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Convolution stride
        padding: Padding applied to the input (applied to left for causal)
        dilation: Dilation factor
        groups: Number of blocked connections between input and output channels

    Returns:
        Output tensor of shape (batch, out_channels, length)
    """
    logger.debug("GEMS CAUSAL_CONV1D")

    # Import conv1d from flag_gems
    from flag_gems.ops.conv1d import conv1d as gems_conv1d

    # Ensure input is 3D
    if input.dim() == 2:
        input = input.unsqueeze(0)  # (in_channels, length) -> (1, in_channels, length)
        squeeze_batch = True
    else:
        squeeze_batch = False

    batch_size, in_channels, length = input.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape

    if groups != 1:
        raise NotImplementedError("Causal convolution with groups > 1 is not supported")

    # For causal convolution, we need left padding only
    # The left padding ensures that output[t] only depends on input[0:t+1]
    # Left padding = dilation * (kernel_size - 1) + padding
    left_padding = dilation * (kernel_size - 1) + padding

    # Apply left padding using torch.nn.functional.pad
    # Pad format is (left, right, front, back) for 3D
    if left_padding > 0:
        # For 3D tensor (batch, channels, length), we pad on the length dimension
        input = torch.nn.functional.pad(input, (left_padding, 0), value=0)

    # Use flag_gems conv1d for the actual convolution
    # Note: We pass 0 as padding since we already padded manually
    output = gems_conv1d(input, weight, bias=bias, stride=stride, padding=0, dilation=dilation, groups=groups)

    # If we started with 2D input, squeeze back
    if squeeze_batch:
        output = output.squeeze(0)

    return output