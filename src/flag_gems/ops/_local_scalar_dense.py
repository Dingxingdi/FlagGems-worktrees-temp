import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _local_scalar_dense_kernel(
    inp,
    out,
    BLOCK_SIZE: tl.constexpr,
):
    # Load only the first element (index 0)
    # Using a mask to ensure we only load the first element
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset == 0  # Only the first position
    inp_val = tl.load(inp + offset, mask=mask, other=0.0)
    # Sum to get just the first element
    result = tl.sum(inp_val, axis=0)
    tl.store(out, result)


def _local_scalar_dense(inp: torch.Tensor):
    logger.debug("GEMS _LOCAL_SCALAR_DENSE")
    # For empty tensors, raise an error
    if inp.numel() == 0:
        raise RuntimeError("cannot call _local_scalar_dense on a tensor with no elements")

    # Make input contiguous for efficient access
    inp = inp.contiguous()

    block_size = 1  # We only need to load one element

    # Create a 0-d output tensor
    out = torch.empty([], dtype=inp.dtype, device=inp.device)

    # Launch kernel with single thread to read first element
    _local_scalar_dense_kernel[(1,)](
        inp,
        out,
        block_size,
    )

    # Convert tensor to Python scalar
    # Copy to CPU first to avoid recursion when calling .item()
    out_cpu = out.to("cpu")
    result = out_cpu.item()
    return result