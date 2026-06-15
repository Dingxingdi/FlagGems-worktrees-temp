import logging

import torch

logger = logging.getLogger(__name__)


def is_strides_like_format(inp: torch.Tensor, memory_format: torch.memory_format) -> bool:
    r"""Check if the tensor's strides match the given memory format.

    Args:
        inp: Input tensor
        memory_format: Memory format to check against (contiguous_format, channels_last, or channels_last_3d)

    Returns:
        bool: True if the tensor's strides match the given memory format
    """
    logger.debug("GEMS IS_STRIDES_LIKE_FORMAT")
    return inp.is_contiguous(memory_format=memory_format)