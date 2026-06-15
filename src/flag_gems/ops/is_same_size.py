import logging

import torch

logger = logging.getLogger(__name__)


def is_same_size(input: torch.Tensor, other: torch.Tensor) -> bool:
    """Check if two tensors have the same size (shape)."""
    logger.debug("GEMS IS_SAME_SIZE")
    return input.shape == other.shape