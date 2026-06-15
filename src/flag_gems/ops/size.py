import logging

import torch

logger = logging.getLogger(__name__)


def size(inp, dim=None):
    """Returns the size of the input tensor.

    Args:
        inp: Input tensor
        dim: Optional dimension to get size of

    Returns:
        If dim is None, returns torch.Size (list of dimensions)
        If dim is specified, returns int (size of that dimension)
    """
    logger.debug("GEMS SIZE")
    if dim is None:
        return inp.size()
    else:
        return inp.size(dim=dim)