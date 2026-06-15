import logging

import torch
from flag_gems import runtime

logger = logging.getLogger(__name__)


def _list_to_tensor(self):
    """Convert a Python list of integers to a tensor.

    This operator takes a Python list of integers and converts it to a 1D tensor
    with int32 dtype on the target device.
    """
    logger.debug("GEMS _list_to_tensor")
    # Convert Python list to tensor with int32 dtype on the target device
    if isinstance(self, list):
        result = torch.tensor(self, dtype=torch.int32, device=runtime.device.name)
    else:
        # Handle case where input might already be a list-like
        result = torch.tensor(list(self), dtype=torch.int32, device=runtime.device.name)

    return result