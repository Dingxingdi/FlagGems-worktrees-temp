import logging

import torch

from flag_gems.ops.add import add
from flag_gems.ops.softmax import softmax

logger = logging.getLogger(__name__)


def add_softmax(input1, input2, dim=-1):
    """Fused add + softmax operation.

    Computes: softmax(input1 + input2, dim)

    This is a convenience function that combines add and softmax operations.
    Both operations use Triton kernels for GPU acceleration.

    Args:
        input1: First input tensor
        input2: Second input tensor (must have same shape as input1)
        dim: Dimension along which to compute softmax (default: -1)

    Returns:
        Tensor with softmax(input1 + input2, dim) applied
    """
    logger.debug("GEMS ADD_SOFTMAX")

    # Validate input shapes
    assert input1.shape == input2.shape, "Input shapes must match"

    # Compute add first using flag_gems.add (Triton kernel)
    added = add(input1, input2)

    # Then apply softmax using flag_gems.softmax (Triton kernel)
    return softmax(added, dim)