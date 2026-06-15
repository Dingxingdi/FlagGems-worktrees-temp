import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _reshape_alias(self: Tensor, size, stride):
    logger.debug("GEMS _RESHAPE_ALIAS")
    """
    Wrapper for aten::_reshape_alias
    Returns a view of the tensor with the given size and stride.
    """
    # Use as_strided to create a view with custom size and stride
    # This avoids calling torch.ops.aten._reshape_alias which would cause recursion
    return self.as_strided(size, stride)