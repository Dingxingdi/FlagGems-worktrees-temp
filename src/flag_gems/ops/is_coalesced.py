import logging

import torch

logger = logging.getLogger(__name__)


def is_coalesced(A: torch.Tensor) -> bool:
    logger.debug("GEMS IS_COALESCED")
    if A.layout != torch.sparse_coo:
        raise RuntimeError(
            f"is_coalesced expected sparse coordinate tensor layout but got {A.layout}"
        )
    return A.is_coalesced()