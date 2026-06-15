import logging

import torch

from flag_gems.ops.cumsum import cumsum as cumsum_impl
from flag_gems.ops.cumsum import cumsum_wrapper

logger = logging.getLogger(__name__)


def cumsum_(inp, dim, *, dtype=None):
    logger.debug("GEMS CUMSUM_")
    result = cumsum_wrapper(inp, dim, dtype)
    inp.copy_(result)
    return inp