import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("elementwise_generic"))
@triton.jit
def nonzero_static_kernel(
    inp,
    prefix_sum,
    out,
    n_elements,
    shape,
    ndim: tl.constexpr,
    size: tl.constexpr,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask).to(tl.int1)
    out_offset = tl.load(prefix_sum + offset, mask=mask) - 1

    nonzero_mask = mask and inp_vals

    idx_flat = offset
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        # Only write if out_offset < size (to handle case where actual nonzero count > size)
        write_mask = nonzero_mask and (out_offset < size)
        tl.store(out + out_offset * ndim + dim, remainder, mask=write_mask)


@libentry()
@triton.jit
def nonzero_static_fill_kernel(
    out,
    ndim: tl.constexpr,
    start_idx: tl.constexpr,
    size: tl.constexpr,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    write_offset = start_idx + offset
    mask = write_offset < size

    for dim in range(ndim):
        tl.store(out + write_offset * ndim + dim, fill_value, mask=mask)


def nonzero_static(inp, *, size, fill_value=-1):
    logger.debug("GEMS NONZERO_STATIC")

    inp_ndim = inp.ndim
    size = int(size)
    fill_value = int(fill_value)

    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)

    shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)

    inp_bool = inp_view
    if inp_view.dtype != torch.bool:
        inp_bool = inp_view != 0

    prefix_sum = inp_bool.cumsum(axis=0)

    num_nonzeros = n_elements
    # Output size is fixed to 'size'
    out = torch.empty((size, inp_ndim), dtype=torch.int64, device=inp.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(inp.device):
        nonzero_static_kernel[grid](
            inp_bool, prefix_sum, out, n_elements, shape, inp_ndim, size, fill_value
        )

    actual_nonzeros = prefix_sum[n_elements - 1].item() if n_elements > 0 else 0

    # Fill remaining positions with fill_value
    if actual_nonzeros < size:
        remaining = size - actual_nonzeros
        fill_grid = lambda meta: (triton.cdiv(remaining, meta["BLOCK_SIZE"]),)
        nonzero_static_fill_kernel[fill_grid](
            out, inp_ndim, actual_nonzeros, size, fill_value, BLOCK_SIZE=1024
        )

    return out