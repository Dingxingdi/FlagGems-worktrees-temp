import logging

import torch

logger = logging.getLogger(__name__)


def sym_constrain_range(size, min=None, max=None):
    """Constrain symbolic shape value for torch.compile.

    This operator is a compiler hint that doesn't perform any actual computation.
    It returns None as per the PyTorch schema.

    Args:
        size: A scalar value representing the symbolic shape to constrain
        min: Optional minimum bound
        max: Optional maximum bound

    Returns:
        None (void)
    """
    logger.debug("GEMS sym_constrain_range")
    # This is a no-op for symbolic shape constraint
    # It doesn't perform any actual computation
    return None


def sym_constrain_range_for_size(size, min=None, max=None):
    """Constrain symbolic shape value for size computation.

    This operator is a compiler hint that doesn't perform any actual computation.
    It returns None as per the PyTorch schema.

    Args:
        size: A scalar value representing the symbolic shape to constrain
        min: Optional minimum bound
        max: Optional maximum bound

    Returns:
        None (void)
    """
    logger.debug("GEMS sym_constrain_range_for_size")
    # This is a no-op for symbolic shape constraint
    # It doesn't perform any actual computation
    return None