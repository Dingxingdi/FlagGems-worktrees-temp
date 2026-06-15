import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


class ConvGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        logger.debug("GEMS CONV+GELU FORWARD")
        # Import here to avoid circular imports
        from flag_gems.ops import conv2d as gems_conv2d
        from flag_gems.ops import gelu as gems_gelu

        # First run conv2d
        conv_out = gems_conv2d(input, weight, bias, stride, padding, dilation, groups)
        # Then apply gelu
        gelu_out = gems_gelu(conv_out)

        # Save for backward
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        # Save references to the functions for backward
        ctx.gems_conv2d = gems_conv2d
        ctx.gems_gelu = gems_gelu

        return gelu_out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS CONV+GELU BACKWARD")
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        gems_conv2d = ctx.gems_conv2d

        # For backward, we need to compute gradients through conv2d
        # The gradient of gelu has already been applied to grad_output by PyTorch's autograd

        # We need to manually compute gradient of conv2d
        # Using torch autograd with a hook approach
        input.requires_grad = True
        weight.requires_grad = True

        # Create a wrapper function that returns conv output
        def conv_func(inp, w, b):
            return gems_conv2d(inp, w, b, stride, padding, dilation, groups)

        # Use torch.autograd.grad to get gradients
        grad_inputs = torch.autograd.grad(
            outputs=conv_func(input, weight, bias),
            inputs=[input, weight],
            grad_outputs=grad_output,
            create_graph=False,
            allow_unused=False
        )

        grad_input = grad_inputs[0]
        grad_weight = grad_inputs[1]

        # Handle bias gradient if needed
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None


def conv_gelu(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Applies a 2D convolution followed by GELU activation.

    Args:
        input: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in, kH, kW)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride of the convolution
        padding: Padding applied to the input
        dilation: Dilation of the convolution
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    return ConvGELU.apply(input, weight, bias, stride, padding, dilation, groups)