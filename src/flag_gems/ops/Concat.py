import logging
from typing import List, Tuple, Union

import torch

from flag_gems.ops.cat import cat as cat_impl

logger = logging.getLogger(__name__)


def concatenate(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    """Concatenates tensors along a given dimension.

    This is a wrapper around the cat implementation that provides the
    aten::concatenate operation.

    Args:
        A: Sequence of tensors to concatenate
        dim: Dimension along which to concatenate

    Returns:
        Concatenated tensor
    """
    logger.debug("GEMS CONCATENATE")
    return cat_impl(A, dim)


def concat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    """Concatenates tensors along a given dimension.

    Alias for concatenate.

    Args:
        A: Sequence of tensors to concatenate
        dim: Dimension along which to concatenate

    Returns:
        Concatenated tensor
    """
    logger.debug("GEMS CONCAT")
    return cat_impl(A, dim)