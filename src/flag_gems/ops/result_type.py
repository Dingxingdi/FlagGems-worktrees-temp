import logging

import torch

logger = logging.getLogger(__name__)


def result_type(input, other):
    """Returns the dtype that would result from performing an arithmetic operation
    on the provided input tensors.

    This is a wrapper around torch.result_type that integrates with FlagGems.
    """
    logger.debug("GEMS RESULT_TYPE")
    return torch.result_type(input, other)