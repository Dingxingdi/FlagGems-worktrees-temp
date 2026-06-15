import logging

import torch

from flag_gems.ops.layernorm import layer_norm

logger = logging.getLogger(__name__)


def LayerNorm(input, normalized_shape=None, weight=None, bias=None, eps=1e-5):
    """LayerNorm operator.

    This is a simplified interface that uses the last dimension of the input
    as the normalized_shape if not provided.

    Args:
        input: Input tensor to normalize
        normalized_shape: Shape to normalize over (optional, defaults to last dim)
        weight: Optional weight tensor
        bias: Optional bias tensor
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    logger.debug("GEMS LAYERNORM")

    # If normalized_shape is not provided, infer from the last dimension
    if normalized_shape is None:
        normalized_shape = (input.shape[-1],)

    # Call the existing layer_norm function
    # layer_norm returns (output, mean, rstd), we only need the output
    output, _, _ = layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

    return output