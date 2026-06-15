import logging
from typing import List

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def unsafe_split_with_sizes(
    self: torch.Tensor,
    split_sizes: List[int],
    dim: int = 0,
):
    """
    Split a tensor into chunks along a given dimension.

    This is the unsafe version of split_with_sizes that skips bounds checking.
    For FlagGems, we delegate to PyTorch's implementation since this is
    fundamentally a view operation that doesn't require data copying.
    """
    logger.debug("GEMS UNSAFE_SPLIT_WITH_SIZES")
    return torch.ops.aten.unsafe_split_with_sizes.default.redispatch(
        _FALLBACK_KEYSET,
        self,
        split_sizes,
        dim,
    )