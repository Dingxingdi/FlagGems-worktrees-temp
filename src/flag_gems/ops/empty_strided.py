import logging

import torch

from flag_gems.runtime import device as device_

logger = logging.getLogger(__name__)


def empty_strided(
    size,
    stride,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=None,
):
    """Create a tensor with the specified size and stride.

    This is a memory allocation operation - the data is undefined.
    """
    logger.debug("GEMS EMPTY_STRIDED")
    if device is None:
        device = torch.device(device_.name)
    return torch.empty_strided(
        size,
        stride,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )