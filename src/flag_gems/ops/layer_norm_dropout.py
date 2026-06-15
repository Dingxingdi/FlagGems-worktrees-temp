import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.dropout import dropout
from flag_gems.ops.layernorm import layer_norm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def layer_norm_dropout(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    dropout_p=0.0,
    eps=1e-5,
    train=True,
):
    """
    Apply Layer Normalization followed by Dropout.

    This is a fused operation that applies LayerNorm first, then Dropout.

    Args:
        input: Input tensor
        normalized_shape: Shape to normalize over
        weight: Optional weight for LayerNorm
        bias: Optional bias for LayerNorm
        dropout_p: Dropout probability
        eps: Epsilon for LayerNorm numerical stability
        train: Whether to apply dropout (True) or not (False)

    Returns:
        output: The result after LayerNorm and Dropout
    """
    logger.debug("GEMS LAYERNORM_DROPOUT FORWARD")

    # Apply LayerNorm first
    y, mean, rstd = layer_norm(input, normalized_shape, weight, bias, eps)

    # Apply Dropout if training and p > 0
    if train and dropout_p > 0:
        y, mask = dropout(y, dropout_p, train)
        return y, mean, rstd, mask
    else:
        # Return a dummy mask for inference mode
        mask = torch.ones_like(y, dtype=torch.bool)
        return y, mean, rstd, mask


def layer_norm_dropout_(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    dropout_p=0.0,
    eps=1e-5,
    train=True,
):
    """
    In-place version of LayerNorm+Dropout (not supported, calls non-inplace version).
    """
    logger.debug("GEMS LAYERNORM_DROPOUT_ FORWARD")
    return layer_norm_dropout(
        input,
        normalized_shape,
        weight=weight,
        bias=bias,
        dropout_p=dropout_p,
        eps=eps,
        train=train,
    )