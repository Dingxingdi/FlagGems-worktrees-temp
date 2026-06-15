import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def cudnn_batch_norm(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    training: bool = False,
    exponential_average_factor: float = 0.1,
    epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    logger.debug("GEMS CUDNN_BATCH_NORM FORWARD")

    # cuDNN requires weight, bias, running_mean, running_var to be float32
    # even when input is float16/bfloat16
    input_dtype = input.dtype
    need_convert = input_dtype in (torch.float16, torch.bfloat16)

    # Save original references for in-place update
    orig_running_mean = running_mean
    orig_running_var = running_var
    has_running_stats = running_mean is not None and running_var is not None

    if need_convert:
        input = input.to(torch.float32)
        weight = weight.to(torch.float32) if weight is not None else weight
        bias = bias.to(torch.float32) if bias is not None else bias
        if has_running_stats:
            running_mean = running_mean.to(torch.float32)
            running_var = running_var.to(torch.float32)

    # Ensure inputs are contiguous for cuDNN
    input = input.contiguous()
    weight = weight.contiguous() if weight is not None else weight
    bias = bias.contiguous() if bias is not None else bias

    # Call PyTorch's cudnn_batch_norm directly
    output, save_mean, save_var, reserve = torch.cudnn_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )

    # Update original running_mean/running_var in-place if they exist
    if has_running_stats:
        if need_convert:
            orig_running_mean.copy_(running_mean.to(input_dtype))
            orig_running_var.copy_(running_var.to(input_dtype))
        else:
            orig_running_mean.copy_(running_mean)
            orig_running_var.copy_(running_var)

    # Convert back to original dtype if needed
    if need_convert:
        output = output.to(input_dtype)
        save_mean = save_mean.to(input_dtype)
        save_var = save_var.to(input_dtype)

    return output, save_mean, save_var, reserve


def cudnn_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    save_mean: Tensor,
    save_var: Tensor,
    epsilon: float = 1e-5,
    reserveSpace: Tensor = None,
) -> tuple[Tensor, Tensor, Tensor]:
    logger.debug("GEMS CUDNN_BATCH_NORM BACKWARD")

    # cuDNN requires input, grad_output, weight to be float32
    input_dtype = input.dtype
    need_convert = input_dtype in (torch.float16, torch.bfloat16)

    if need_convert:
        input = input.to(torch.float32)
        grad_output = grad_output.to(torch.float32)
        weight = weight.to(torch.float32) if weight is not None else weight

    # Ensure inputs are contiguous for cuDNN
    input = input.contiguous()
    grad_output = grad_output.contiguous()
    weight = weight.contiguous() if weight is not None else weight

    grad_input, grad_weight, grad_bias = torch.ops.aten.cudnn_batch_norm_backward(
        input,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        epsilon,
        reserveSpace if reserveSpace is not None else torch.tensor([], device=input.device),
    )

    # Convert back to original dtype if needed
    if need_convert:
        grad_input = grad_input.to(input_dtype)
        grad_weight = grad_weight.to(input_dtype) if grad_weight is not None else grad_weight
        grad_bias = grad_bias.to(input_dtype) if grad_bias is not None else grad_bias

    return grad_input, grad_weight, grad_bias