import logging

from flag_gems.ops.argmin import argmin

logger = logging.getLogger(__name__)


def ReduceArgMin(inp, dim=None, keepdim=False, *, dtype=None):
    """ReduceArgMin operator - returns indices of the minimum values.

    This is an alias for argmin functionality.
    """
    logger.debug("GEMS ReduceArgMin")
    return argmin(inp, dim=dim, keepdim=keepdim, dtype=dtype)