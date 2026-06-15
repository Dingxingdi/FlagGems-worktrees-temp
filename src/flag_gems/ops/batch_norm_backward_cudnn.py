import logging
import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.ops.batch_norm import (
    batch_norm_backward as native_batch_norm_backward,
    make_3d_for_bn,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    save_mean: Tensor,
    save_var: Tensor,
    update: bool,
    eps: float,
    output_mask: list,
    reserve: Tensor,
):
    """
    Batch normalization backward operator (cuDNN version).

    This is a wrapper around native_batch_norm_backward that converts
    the cuDNN interface (save_var, update) to the native interface
    (save_invstd, train).

    Args:
        grad_out: Gradient of the output
        input: Input tensor
        weight: Weight tensor (or None)
        running_mean: Running mean (or None)
        running_var: Running variance (or None)
        save_mean: Saved mean from forward pass
        save_var: Saved variance from forward pass
        update: Whether to update running statistics (training mode)
        eps: Epsilon for numerical stability
        output_mask: Which outputs to return [grad_input, grad_weight, grad_bias]
        reserve: Reserve space (not used in computation)

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias) based on output_mask
    """
    logger.debug("GEMS BATCH_NORM_BACKWARD")

    # Convert save_var to save_invstd (inverse standard deviation)
    # save_invstd = 1 / sqrt(save_var + eps)
    save_invstd = torch.rsqrt(save_var + eps)

    # The 'update' parameter is similar to 'train' in native_batch_norm_backward
    # In training mode, update=True; in inference mode, update=False
    train = update

    # Call the native implementation
    return native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    )