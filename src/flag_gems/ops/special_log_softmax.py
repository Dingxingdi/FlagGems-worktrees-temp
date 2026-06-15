import logging

import torch

from flag_gems.ops.log_softmax import log_softmax

logger = logging.getLogger(__name__)


def special_log_softmax(self, dim, half_to_float=False):
    """Special log softmax function.

    This is an alias for torch.special.log_softmax which is mathematically
    equivalent to log_softmax.
    """
    logger.debug("GEMS SPECIAL_LOG_SOFTMAX")
    return log_softmax(self, dim, half_to_float)