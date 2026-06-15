import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def sym_constrain_range_for_size(
    size: Any,
    *,
    min: Optional[int] = None,
    max: Optional[int] = None,
):
    """
    A no-op operator for symbolic shape validation in torch.compile.
    This operator validates that a symbolic size is within a certain range.
    In practice, it does nothing - it's a side-effect-only operation used
    by the compiler for shape constraint validation.
    """
    logger.debug("GEMS SYM_CONSTRAIN_RANGE_FOR_SIZE")

    # Convert size to int if it's a tensor
    if isinstance(size, torch.Tensor):
        size_val = size.item()
    else:
        size_val = size

    # Perform validation if min/max are specified
    if min is not None and size_val < min:
        raise RuntimeError(f"Invalid value range for {size_val} between [{min}, {max}].")
    if max is not None and size_val > max:
        raise RuntimeError(f"Invalid value range for {size_val} between [{min}, {max}].")

    # This is a no-op - it returns nothing (void)
    # The validation is handled at compile time by torch.compile
    return None