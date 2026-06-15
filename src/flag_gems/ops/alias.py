import logging

import torch

logger = logging.getLogger(__name__)


def alias(inp: torch.Tensor):
    logger.debug("GEMS ALIAS")
    """
    Wrapper for aten::alias
    Returns a view of the input tensor.
    Uses detach() to create a new tensor that shares the same storage
    but is a separate view for dispatch purposes.
    """
    return inp.detach()