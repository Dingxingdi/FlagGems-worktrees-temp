import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def relu_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output = tl.where(input > 0, input, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


class Conv2dActivation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        logger.debug("GEMS CONV2D_ACTIVATION FORWARD")

        # Import flag_gems conv2d here to avoid circular imports
        import flag_gems

        # Call flag_gems conv2d
        conv_out = flag_gems.conv2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        # Apply ReLU activation using triton kernel
        output = conv_out.contiguous()
        n_elements = output.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        relu_kernel[grid](
            output,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(weight, input, bias)
        ctx.stride = stride if isinstance(stride, tuple) else (stride, stride)
        ctx.padding = padding if isinstance(padding, tuple) else (padding, padding)
        ctx.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        ctx.groups = groups

        return output


def conv_activation(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """
    Fused Conv2d + ReLU activation.

    This is a fused operator that performs 2D convolution followed by ReLU activation.

    Args:
        input: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, kH, kW)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for the convolution
        padding: Padding for the convolution
        dilation: Dilation for the convolution
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    return Conv2dActivation.apply(input, weight, bias, stride, padding, dilation, groups)