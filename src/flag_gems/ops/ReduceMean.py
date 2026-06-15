import logging

import torch

# Import the actual Triton implementations from mean.py
from flag_gems.ops.mean import mean, mean_dim

logger = logging.getLogger(__name__)


def ReduceMean(inp, dim=None, keepdim=False, *, dtype=None):
    """ReduceMean operator - computes the mean of tensor elements.

    This is an alias for torch.mean functionality in FlagGems.
    The computation is performed by Triton kernels (via mean/mean_dim).
    """
    logger.debug("GEMS ReduceMean")
    if dim is None:
        return mean(inp, dtype=dtype)
    else:
        return mean_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype)


def ReduceMean_(inp, dim=None, keepdim=False, *, dtype=None):
    """In-place ReduceMean operator (not supported, but required for API compatibility)."""
    logger.debug("GEMS ReduceMean_")
    raise RuntimeError("ReduceMean_ (in-place) is not supported for ReduceMean")