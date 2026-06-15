import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def unsqueeze(A: torch.Tensor, dim: int) -> torch.Tensor:
    """Functional unsqueeze operation: adds a dimension of size 1 at the specified dim."""
    logger.debug("GEMS UNSQUEEZE")

    # Validate and normalize dim
    ndim = A.ndim
    assert dim >= -ndim - 1 and dim <= ndim, (
        f"Dimension out of range (expected to be in range of [{-ndim - 1}, {ndim}], "
        f"but got {dim})"
    )
    # Normalize negative dim
    if dim < 0:
        dim = dim + ndim + 1

    # Use redispatch to bypass flag_gems registration and avoid recursion
    return torch.ops.aten.unsqueeze.default.redispatch(_FALLBACK_KEYSET, A, dim)


def unsqueeze_(A: torch.Tensor, dim: int) -> torch.Tensor:
    """In-place unsqueeze operation: adds a dimension of size 1 at the specified dim."""
    logger.debug("GEMS UNSQUEEZE_")

    # Validate and normalize dim
    ndim = A.ndim
    assert dim >= -ndim - 1 and dim <= ndim, (
        f"Dimension out of range (expected to be in range of [{-ndim - 1}, {ndim}], "
        f"but got {dim})"
    )
    # Normalize negative dim
    if dim < 0:
        dim = dim + ndim + 1

    # Use redispatch to bypass flag_gems registration and avoid recursion
    return torch.ops.aten.unsqueeze_.default.redispatch(_FALLBACK_KEYSET, A, dim)