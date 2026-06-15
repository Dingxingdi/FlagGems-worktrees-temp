import logging

import torch

logger = logging.getLogger(__name__)


def view_as(self, other):
    """
    View this tensor as the same size as :attr:`other`.
    ``self.view_as(other)`` is equivalent to ``self.view(other.size())``.
    """
    logger.debug("GEMS VIEW_AS")
    # Check that the number of elements is the same
    if self.numel() != other.numel():
        raise RuntimeError(
            f"shape '{other.shape}' is invalid for input of size {self.numel()}"
        )
    # Return a view with the shape of other
    return self.view(other.size())