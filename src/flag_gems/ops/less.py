import logging

import torch

import flag_gems
from flag_gems.ops.lt import lt, lt_scalar

logger = logging.getLogger(__name__)


def less(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Performs element-wise less than comparison.

    This is an alias for torch.lt (less than).
    """
    logger.debug("GEMS LESS")
    return lt(A, B)


def less_scalar(A: torch.Tensor, B) -> torch.Tensor:
    """Performs element-wise less than comparison with a scalar.

    This is an alias for torch.lt with scalar operand.
    """
    logger.debug("GEMS LESS SCALAR")
    return lt_scalar(A, B)