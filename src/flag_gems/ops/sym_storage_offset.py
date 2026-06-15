import logging

import torch

logger = logging.getLogger(__name__)


def sym_storage_offset(inp):
    """Return the storage offset of the input tensor.

    This operator returns the offset of the tensor's storage relative to the
    beginning of the storage, measured in number of elements.
    """
    logger.debug("GEMS SYM_STORAGE_OFFSET")
    return inp.storage_offset()