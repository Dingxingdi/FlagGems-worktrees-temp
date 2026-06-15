import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def item_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    inp_val = tl.load(inp + offset, mask=mask)
    tl.store(out + offset, inp_val, mask=mask)


def item(inp):
    """Return the value of this tensor as a standard Python number.

    This only works for tensors with one element.
    """
    logger.debug("GEMS ITEM")
    n_elements = inp.numel()
    if n_elements != 1:
        raise RuntimeError(
            f"item(): tensor has {n_elements} elements, but item() only works "
            "for tensors with one element. For other cases, see tolist()."
        )

    # Create a 0-d output tensor to store the result
    out = torch.empty([], dtype=inp.dtype, device=inp.device)

    # Launch a simple kernel to read the value
    BLOCK_SIZE = 1
    item_kernel[(1,)](inp, out, n_elements, BLOCK_SIZE)

    # Return the Python scalar value
    # Use .cpu().item() to avoid recursion - moving to CPU bypasses our dispatch
    return out.cpu().item()