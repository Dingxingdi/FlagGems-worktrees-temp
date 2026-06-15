import logging

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def _make_dep_token_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Store uninitialized value (using 0.0 as placeholder, actual value doesn't matter)
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def _make_dep_token(
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """
    Create a dependency token for TorchScript tracing.

    This operator creates a 0-dimensional (scalar) tensor that can be used
    for dependency tracking in TorchScript. The value of the returned tensor
    is uninitialized and should not be used.
    """
    logger.debug("GEMS _make_dep_token")
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = flag_gems.device

    # Create output tensor
    out = torch.empty((), dtype=dtype, device=device)

    # Since this is a 0-dimensional tensor (scalar), there's only 1 element
    # We still use the kernel pattern for consistency with FlagGems
    n_elements = 1
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    BLOCK_SIZE = 1
    _make_dep_token_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out