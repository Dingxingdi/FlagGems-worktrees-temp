import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def convolution_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    """
    Determines the output size of a convolution operation.
    """
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv2d_forward"),
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
def conv2d_forward_kernel(
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

    # caculate in_n out_height out_weight value in kernel
    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    # Load the input and weight pointers. input and weight are of shape
    # [in_n, groups, in_c, input_height, input_width] and [groups, out_c, in_c, weight_height, weight_width]
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
    for hwc in range(weight_height * weight_width * BLOCK_CI_COUNT):
        c = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        h = hw // weight_width
        w = hw % weight_width

        input_c_offset = c + tl.arange(0, BLOCK_CI)
        input_height_offset = (
            h * dilation_height
            - padding_height
            + stride_height * output_height_point_value
        )
        input_width_offset = (
            w * dilation_width - padding_width + stride_width * output_width_point_value
        )

        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (0 <= input_height_offset)[:, None]
            & (input_height_offset < input_height)[:, None]
            & (0 <= input_width_offset)[:, None]
            & (input_width_offset < input_width)[:, None]
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
    bias = tl.load(bias_pointer, mask_bias).to(tl.float32)
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


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv2d_backward_weight"),
    key=[
        "in_n",
        "input_height",
        "input_width",
        "weight_height",
        "weight_width",
        "input_c",
        "stride_height",
        "stride_width",
        "out_height",
        "out_width",
        "out_c",
        "padding_height",
        "padding_width",
    ],
)
@triton.jit
def conv2d_backward_kernel_weight(
    input_pointer,
    out_grad_pointer,
    weight_pointer,
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
    input_height,
    input_width,
    weight_height,
    weight_width,
    input_c,
    in_n,
    stride_height,
    stride_width,
    out_height,
    out_width,
    out_c,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    BLOCK_NO: tl.constexpr,
    BLOCK_CI_HK_WK: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # load out_grad n (groups out_c)  ho wo
    # load weight (groups out_c) ci h w
    # load input n (groups ci)  hi wi

    # init pid and offset 0 for ci*hk*wk, 1 for groups, 2 for co.
    pid_ci_hk_wk = tl.program_id(0)
    pid_groups = tl.program_id(1)
    pid_co = tl.program_id(2)

    # caculate ci weight_height weight_weight value in kernel
    ci_hk_wk_offset = pid_ci_hk_wk * BLOCK_CI_HK_WK + tl.arange(0, BLOCK_CI_HK_WK)
    ci_hk_offset = ci_hk_wk_offset // weight_width
    ci_point_value = ci_hk_offset // weight_height
    weight_height_point_value = ci_hk_offset % weight_height
    weight_width_point_value = ci_hk_wk_offset % weight_width

    # caculate init pointer info of tensors
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    out_grad_pointer += (output_c_offset * output_c_stride)[None, :] + (
        pid_groups[None] * output_c_stride * out_c
    )[:, None]

    weight_pointer += (
        pid_groups * weight_n_stride * out_c + output_c_offset * weight_n_stride
    )[None, :] + (
        ci_point_value * weight_c_stride
        + weight_height_point_value * weight_height_stride
        + weight_width_point_value * weight_width_stride
    )[
        :, None
    ]

    input_pointer += (ci_point_value * input_c_stride[None])[:, None] + (
        pid_groups[None] * input_c_stride * input_c
    )[None, :]

    # calculate the values of the input based on the width and height of the output by looping
    accum = tl.zeros((BLOCK_CI_HK_WK, BLOCK_CO), dtype=tl.float32)
    for h in range(0, out_height):
        for w in range(0, out_width):
            for n in range(0, in_n, BLOCK_NO):
                output_n_offset = n + tl.arange(0, BLOCK_NO)

                # caculate input pointer to [cin*kh*kw, *] out_grad pointer to [*, out_c], N*hout*wout as reduce dim
                curr_out_grad_pointer = (
                    out_grad_pointer
                    + (
                        output_n_offset * output_n_stride
                        + h * output_height_stride
                        + w * output_width_stride
                    )[:, None]
                )
                out_grad_mask = (output_n_offset < in_n)[:, None] & (
                    output_c_offset < out_c
                )[None, :]

                curr_out_grad = tl.load(curr_out_grad_pointer, mask=out_grad_mask)

                input_height_offset = (
                    weight_height_point_value * dilation_height
                    - padding_height
                    + stride_height * h
                )

                input_width_offset = (
                    weight_width_point_value * dilation_width
                    - padding_width
                    + stride_width * w
                )

                curr_input_pointer = (
                    input_pointer
                    + (input_n_stride * output_n_offset)[None, :]
                    + (input_height_stride * input_height_offset)[:, None]
                    + (input_width_stride * input_width_offset)[:, None]
                )
                input_mask = (
                    (output_n_offset < in_n)[None, :]
                    & (ci_point_value < input_c)[:, None]
                    & (0 <= input_height_offset)[:, None]
                    & (input_height_offset < input_height)[:, None]
                    & (0 <= input_width_offset)[:, None]
                    & (input_width_offset < input_width)[:, None]
                )

                curr_input = tl.load(curr_input_pointer, mask=input_mask)
                accum += tl.dot(curr_input, curr_out_grad, allow_tf32=False)

    weight_mask = (
        (ci_point_value < input_c)[:, None]
        & (output_c_offset < out_c)[None, :]
        & (weight_height_point_value < weight_height)[:, None]
        & (weight_width_point_value < weight_width)[:, None]
    )
    tl.store(weight_pointer, accum, weight_mask)


class ConvolutionBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    ):
        logger.debug("GEMS CONVOLUTION_BACKWARD")

        # Handle transposed convolution (currently not supported, fall back to torch)
        if transposed:
            return torch.ops.aten.convolution_backward(
                grad_output,
                input,
                weight,
                bias_sizes,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
                output_mask,
            )

        # Parse stride
        if len(stride) == 2:
            stride_height, stride_width = stride
        else:
            stride_height = stride_width = stride[0] if stride else 1

        # Parse padding
        if len(padding) == 2:
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = padding[0] if padding else 0

        # Parse dilation
        if len(dilation) == 2:
            dilation_height, dilation_width = dilation
        else:
            dilation_height = dilation_width = dilation[0] if dilation else 1

        # Get tensor dimensions
        in_n, in_c, input_height, input_width = input.shape
        out_c, weight_c, weight_height, weight_width = weight.shape

        out_height = grad_output.shape[2]
        out_width = grad_output.shape[3]

        # Compute grad_input if requested
        grad_input = None
        if output_mask[0]:
            # Prepare revert weight (flip and transpose)
            revert_weight = weight.clone()
            revert_weight = torch.flip(revert_weight, dims=[2, 3]).contiguous()

            if groups != 1:
                revert_weight = revert_weight.reshape(
                    groups, out_c, weight_c, weight_height, weight_width
                )
                revert_weight = revert_weight.transpose(1, 2)
                revert_weight = revert_weight.reshape(
                    groups * weight_c, out_c, weight_height, weight_width
                ).contiguous()
            else:
                revert_weight = revert_weight.transpose(0, 1).contiguous()

            # Calculate new output dimensions for conv with stride
            new_out_height = out_height + (stride_height - 1) * (out_height - 1)
            new_out_width = out_width + (stride_width - 1) * (out_width - 1)

            # Insert stride in grad_output
            if stride_height > 1 or stride_width > 1:
                new_out = torch.zeros(
                    grad_output.shape[0],
                    grad_output.shape[1],
                    new_out_height,
                    new_out_width,
                    device=grad_output.device,
                    dtype=grad_output.dtype,
                )
                for i in range(out_height):
                    for j in range(out_width):
                        new_out[:, :, i * stride_height, j * stride_width] = grad_output[
                            :, :, i, j
                        ]
            else:
                new_out = grad_output

            revert_padding_height = (
                dilation_height * (weight_height - 1) - padding_height
            )
            revert_padding_width = dilation_width * (weight_width - 1) - padding_width

            grad_input = torch.zeros(
                in_n,
                weight_c * groups,
                input_height,
                input_width,
                dtype=torch.float32,
                device=grad_output.device,
            )

            grid = lambda META: (
                triton.cdiv(
                    grad_output.shape[0] * input_height * input_width,
                    META["BLOCK_NI_HO_WO"],
                ),
                triton.cdiv(int(weight_c), META["BLOCK_CO"]),
                groups,
            )
            bias_zero = torch.zeros(
                groups * weight_c, device=grad_output.device, dtype=grad_output.dtype
            )
            conv2d_forward_kernel[grid](
                new_out,
                revert_weight,
                grad_input,
                bias_zero,
                grad_output.shape[0],
                new_out_height,
                new_out_width,
                groups * weight_c,
                input_height,
                input_width,
                *new_out.stride(),
                *revert_weight.stride(),
                *grad_input.stride(),
                out_c,
                weight_height,
                weight_width,
                1,
                1,
                revert_padding_height,
                revert_padding_width,
                dilation_height,
                dilation_width,
                groups=groups,
            )
            grad_input = grad_input.to(input.dtype)

        # Compute grad_weight if requested
        grad_weight = None
        if output_mask[1]:
            grad_weight = torch.zeros(
                out_c * groups,
                weight_c,
                weight_height,
                weight_width,
                dtype=weight.dtype,
                device=grad_output.device,
            )

            grid_weight = lambda meta: (
                triton.cdiv(
                    weight_c * weight_height * weight_width,
                    meta["BLOCK_CI_HK_WK"],
                ),
                groups,
                triton.cdiv(out_c, meta["BLOCK_CO"]),
            )
            conv2d_backward_kernel_weight[grid_weight](
                input,
                grad_output,
                grad_weight,
                *input.stride(),
                *weight.stride(),
                *grad_output.stride(),
                input_height,
                input_width,
                weight_height,
                weight_width,
                weight_c,
                in_n,
                stride_height,
                stride_width,
                out_height,
                out_width,
                out_c,
                padding_height,
                padding_width,
                dilation_height,
                dilation_width,
            )

        # Compute grad_bias if requested
        grad_bias = None
        if output_mask[2] and bias_sizes is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias


def convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    transposed=False,
    output_padding=(0, 0),
    groups=1,
    output_mask=(True, True, True),
):
    # Ensure inputs are on GPU
    if not grad_output.is_cuda:
        grad_output = grad_output.cuda()
    if not input.is_cuda:
        input = input.cuda()
    if not weight.is_cuda:
        weight = weight.cuda()

    # Convert stride, padding, dilation, output_padding to tuples if they are integers
    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)
    if isinstance(dilation, int):
        dilation = (dilation,)
    if isinstance(output_padding, int):
        output_padding = (output_padding,)

    # Handle 2D convolution (the most common case)
    if input.ndim == 4:
        return ConvolutionBackward.apply(
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
    else:
        # Fall back to PyTorch for other cases
        return torch.ops.aten.convolution_backward(
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )