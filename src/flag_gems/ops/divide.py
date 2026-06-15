import logging

import torch

from flag_gems.ops.div import div_mode, div_mode_

logger = logging.getLogger(__name__)


# Re-export div_mode functions as divide
def divide(A, B, *, rounding_mode=None):
    logger.debug("GEMS DIVIDE")
    return div_mode(A, B, rounding_mode=rounding_mode)


def divide_(A, B, *, rounding_mode=None):
    logger.debug("GEMS DIVIDE_")
    return div_mode_(A, B, rounding_mode=rounding_mode)