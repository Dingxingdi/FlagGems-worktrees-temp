import logging

import torch

logger = logging.getLogger(__name__)


def view_as_real(A):
    logger.debug("GEMS VIEW_AS_REAL")
    if not A.is_complex():
        raise TypeError("view_as_real only supports complex input tensors")
    # Delegate to PyTorch's native implementation
    return torch.view_as_real(A)