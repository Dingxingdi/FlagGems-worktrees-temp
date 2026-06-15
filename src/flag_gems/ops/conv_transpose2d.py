import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def conv_transpose2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    output_padding: int,
) -> int:
    """
    Determines the output size of a 2D transposed convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        dilation: Dilation.
        output_padding: Additional output padding.

    Returns:
        Output size of 2D transposed convolution.
    """
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


# Default configuration for conv_transpose2d
CONV_TRANSPOSE2D_CONFIGS = [
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 32, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 64, "BLOCK_CI": 32},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 64, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 128, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
]


@libentry()
@triton.autotune(
    configs=CONV_TRANSPOSE2D_CONFIGS,
    key=[
        "in_n",
        "weight_c",
        "input_height",
        "input_width",
        "out_c",
        "out_height",
        "out_width",
        "weight_height",
        "weight_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def conv_transpose2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # For transposed convolution:
    # For output position (n, ho, wo), we need to find which input positions contribute.
    # The relationship is: ho = ih * stride - padding + kh * dilation
    # So: ih = (ho + padding - kh * dilation) / stride
    # When stride > 1, the input is "dilated" - there are empty positions between inputs.

    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    # Load the input and weight pointers
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI

    # Iterate over kernel positions
    for kh in range(weight_height):
        for kw in range(weight_width):
            # Calculate the starting input position for this kernel position
            # ih = (ho + padding - kh * dilation) / stride
            # iw = (wo + padding - kw * dilation) / stride
            input_height_offset = (
                output_height_point_value + padding_height - kh * dilation_height
            )
            input_width_offset = (
                output_width_point_value + padding_width - kw * dilation_width
            )

            # When stride > 1, the actual input positions are every stride
            # We need to iterate over input positions with step = stride
            for ic_base in range(0, weight_c, BLOCK_CI):
                ic_offset = ic_base + tl.arange(0, BLOCK_CI)
                input_c_offset = ic_offset

                # Compute actual input positions: divide by stride
                # For transposed conv with stride > 1, there are "gaps" in input
                curr_input_heights = input_height_offset // stride_height
                curr_input_widths = input_width_offset // stride_width

                # Check if positions are valid (divisible by stride)
                height_valid = (input_height_offset % stride_height == 0) & (
                    curr_input_heights >= 0
                ) & (curr_input_heights < input_height)
                width_valid = (input_width_offset % stride_width == 0) & (
                    curr_input_widths >= 0
                ) & (curr_input_widths < input_width)
                valid = height_valid & width_valid

                curr_input_pointer = (
                    input_pointer
                    + (input_c_stride * input_c_offset)[None, :]
                    + (input_height_stride * curr_input_heights)[:, None]
                    + (input_width_stride * curr_input_widths)[:, None]
                )
                curr_weight_pointer = (
                    weight_pointer
                    + (weight_c_stride * input_c_offset)[:, None]
                    + (weight_height_stride * kh)
                    + (weight_width_stride * kw)
                )

                input_mask = (
                    (in_n_point_value < in_n)[:, None]
                    & (input_c_offset < weight_c)[None, :]
                    & valid[:, None]
                )
                weight_mask = (input_c_offset < weight_c)[:, None] & (
                    output_c_offset < out_per_group_c
                )[None, :]

                input_block = tl.load(curr_input_pointer, mask=input_mask)
                weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

                accum += tl.dot(input_block, weight_block, allow_tf32=False)

    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask=mask_bias).to(tl.float32)
    accum += bias

    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


class ConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    ):
        logger.debug("GEMS CONVTRANSPOSE2D")

        assert weight.ndim == 4, f"Weights must be 4D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), f"Bias must be 1D, received shape {bias.shape}"

        # For conv_transpose2d:
        # Input: (batch, in_channels, H, W)
        # Weight: (in_channels, out_channels/groups, kH, kW)
        in_channels = weight.shape[0]
        out_channels = weight.shape[1] * groups

        assert (
            input.shape[1] == in_channels
        ), f"Incompatible input channels {input.shape[1]} and weight in_channels {in_channels}"
        assert (
            bias is None or bias.shape[0] == out_channels
        ), f"Incompatible weight out_channels {out_channels} and bias {bias.shape[0]}"

        if isinstance(stride, (list, tuple)):
            stride_height, stride_width = stride
        else:
            stride_height = stride_width = stride

        if isinstance(padding, (list, tuple)):
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = padding

        if isinstance(output_padding, (list, tuple)):
            output_padding_height, output_padding_width = output_padding
        else:
            output_padding_height = output_padding_width = output_padding

        if isinstance(dilation, (list, tuple)):
            dilation_height, dilation_width = dilation
        else:
            dilation_height = dilation_width = dilation

        in_n, _, input_height, input_width = input.shape
        _, out_c_per_group, weight_height, weight_width = weight.shape
        out_c = out_c_per_group * groups

        out_height = conv_transpose2d_output_size(
            input_height,
            weight_height,
            stride_height,
            padding_height,
            dilation_height,
            output_padding_height,
        )
        out_width = conv_transpose2d_output_size(
            input_width,
            weight_width,
            stride_width,
            padding_width,
            dilation_width,
            output_padding_width,
        )

        output_dtype = input.dtype
        output = torch.empty(
            (in_n, out_c, out_height, out_width),
            device=input.device,
            dtype=output_dtype,
        )

        # For conv_transpose2d:
        # Weight is of shape (in_channels, out_c_per_group, kH, kW)
        # We need to transpose to (out_c_per_group, in_channels, kH, kW)
        weight_transposed = weight.transpose(0, 1).contiguous()

        grid = lambda META: (
            triton.cdiv(in_n * out_height * out_width, META["BLOCK_NI_HO_WO"]),
            triton.cdiv(int(out_c // groups), META["BLOCK_CO"]),
            groups,
        )

        if bias is None:
            bias_pointer = torch.zeros(out_c, device=input.device, dtype=output_dtype)
        else:
            bias_pointer = bias

        conv_transpose2d_forward_kernel[grid](
            input,
            weight_transposed,
            output,
            bias_pointer,
            in_n,
            input_height,
            input_width,
            out_c,
            out_height,
            out_width,
            *input.stride(),
            *weight_transposed.stride(),
            *output.stride(),
            in_channels,
            weight_height,
            weight_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
            groups=groups,
        )

        ctx.save_for_backward(weight)
        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.output_padding = (output_padding_height, output_padding_width)
        ctx.dilation = (dilation_height, dilation_width)
        ctx.weight_info = (in_channels, out_c_per_group, weight_height, weight_width)
        ctx.input_info = (in_n, input_height, input_width, out_c)
        ctx.out_info = (out_height, out_width)
        ctx.device = input.device
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS CONVTRANSPOSE2D VJP")
        (weight,) = ctx.saved_tensors

        in_channels, out_c_per_group, weight_height, weight_width = ctx.weight_info
        in_n, input_height, input_width, out_c = ctx.input_info
        out_height, out_width = ctx.out_info

        device = ctx.device
        groups = ctx.groups

        stride_height, stride_width = ctx.stride
        dilation_height, dilation_width = ctx.dilation
        padding_height, padding_width = ctx.padding

        # For backward of transposed conv:
        # Input gradient: regular conv with:
        # - weight: transposed (but not flipped)
        # - stride: 1
        # - padding: dilation * (kernel - 1) - original_padding
        revert_padding_height = dilation_height * (weight_height - 1) - padding_height
        revert_padding_width = dilation_width * (weight_width - 1) - padding_width

        # Weight needs to be transposed for backward
        revert_weight = weight.transpose(0, 1).contiguous()

        # Compute input gradient using conv2d
        from flag_gems.ops.conv2d import conv2d

        input_back = conv2d(
            out_grad,
            revert_weight,
            bias=None,
            stride=1,
            padding=max(revert_padding_height, 0),
            dilation=dilation_height,
            groups=groups,
        )

        # For weight gradient, we compute via conv between input and output_grad
        # Simplified implementation using PyTorch for now
        weight_back = torch.zeros(
            in_channels,
            out_c_per_group,
            weight_height,
            weight_width,
            dtype=weight.dtype,
            device=device,
        )

        # Use manual computation for weight gradient
        for n in range(in_n):
            for g in range(groups):
                in_c_start = g * in_channels // groups
                in_c_end = (g + 1) * in_channels // groups
                out_c_start = g * out_c_per_group
                out_c_end = (g + 1) * out_c_per_group

                for ic in range(in_c_start, in_c_end):
                    for oc in range(out_c_start, out_c_end):
                        for kh in range(weight_height):
                            for kw in range(weight_width):
                                ih_start = kh * dilation_height - padding_height
                                iw_start = kw * dilation_width - padding_width

                                for ih in range(input_height):
                                    oh = ih * stride_height + ih_start
                                    for iw in range(input_width):
                                        ow = iw * stride_width + iw_start

                                        if 0 <= oh < out_height and 0 <= ow < out_width:
                                            weight_back[ic, oc - out_c_start, kh, kw] += (
                                                input[n, ic, ih, iw] * out_grad[n, oc, oh, ow]
                                            )

        if bias is not None:
            bias_grad = out_grad.sum(dim=(0, 2, 3))
        else:
            bias_grad = None

        return (
            input_back,
            weight_back,
            bias_grad,
            None,
            None,
            None,
            None,
            None,
        )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups=1,
):
    return ConvTranspose2d.apply(
        input, weight, bias, stride, padding, output_padding, dilation, groups
    )