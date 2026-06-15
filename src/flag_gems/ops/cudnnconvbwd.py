import logging

import torch

logger = logging.getLogger(__name__)


def cudnnconvbwd(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """Convolution backward pass.

    This function computes the gradient of the convolution operation
    by delegating to torch.ops.aten.convolution_backward, which uses
    cuDNN when available for optimized performance.

    Args:
        grad_output: Gradient of the output (should have same dtype as input)
        input: Input tensor
        weight: Weight tensor
        stride: Stride value
        padding: Padding value
        dilation: Dilation value
        groups: Number of groups

    Returns:
        Tuple of (input_grad, weight_grad, bias_grad)
    """
    logger.debug("GEMS CUDNNCONVBWD")

    # Convert scalar params to lists for aten op
    if isinstance(stride, (int,)):
        stride = [stride, stride]
    if isinstance(padding, (int,)):
        padding = [padding, padding]
    if isinstance(dilation, (int,)):
        dilation = [dilation, dilation]

    # Compute all three gradients
    output_mask = [True, True, True]

    result = torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,  # bias_sizes
        stride,
        padding,
        dilation,
        False,  # transposed
        [0, 0],  # output_padding
        groups,
        output_mask,
    )

    return result