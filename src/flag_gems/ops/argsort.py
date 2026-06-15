import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.sort import (
    convert_to_uint_preverse_order,
    compute_global_hist_kernel,
    sweep,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def radix_sort_indices(arr, k_bits=8, descending=False):
    """Radix sort that returns only sorted indices."""
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_for_global_hist = (m * grid_n, 1, 1)

    with torch_device_fn.device(arr.device):
        global_hist = torch.zeros(
            (m, n_passes, num_bins), device=arr.device, dtype=torch.int32
        )
        compute_global_hist_kernel[grid_for_global_hist](
            arr,
            global_hist,
            n_passes,
            m,
            n,
            tiles_n_per_cta,
            TILE_N,
            TILE_R,
            k_bits,
            descending,
        )
        ex_cumsum_bins = torch.cumsum(global_hist, -1) - global_hist
        ex_cumsum_bins = ex_cumsum_bins.to(torch.uint32)

        # Only need indices, not values
        indices_in = (
            torch.arange(0, n, dtype=torch.int64, device=arr.device)
            .broadcast_to(arr.shape)
            .contiguous()
        )
        indices_out = torch.empty_like(indices_in)

        TILE_R = 8
        grid_r = triton.cdiv(num_bins, TILE_R)
        TILE_N = 2048
        grid_n = triton.cdiv(n, TILE_N)
        grid_for_sweep = (m * grid_n, grid_r)

        # Dummy buffer for values (not used)
        arr_in = torch.clone(arr)
        arr_out = torch.empty_like(arr)

        status = torch.empty(
            (m, num_bins, grid_n), device=arr.device, dtype=torch.uint32
        )

        for i in range(0, n_passes):
            bit_offset = i * k_bits
            status.zero_()
            sweep[grid_for_sweep](
                arr_in,
                indices_in,
                arr_out,
                indices_out,
                ex_cumsum_bins,
                status,
                n_passes,
                i,
                bit_offset,
                m,
                n,
                grid_n,
                TILE_N,
                TILE_R,
                k_bits,
                descending,
            )
            # Swap buffers
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return indices_in


def argsort(inp, dim=-1, descending=False, stable=False):
    """Returns the indices that sort a tensor along a given dimension in ascending order by value.

    This is the second value returned by :meth:`torch.sort`. See its documentation
    for the exact semantics of this method.

    Args:
        input (Tensor): the input tensor.
        dim (int, optional): the dimension to sort along
        descending (bool, optional): controls the sorting order (ascending or descending)
        stable (bool, optional): controls the relative order of equivalent elements

    Returns:
        Tensor: the indices that sort the tensor
    """
    logger.debug("GEMS ARGSORT")
    _ = stable  # We only implement stable radix sort here

    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return torch.zeros_like(inp, dtype=torch.int64)

    # Handle float16 and bfloat16 by converting to float32 for sorting
    orig_dtype = inp.dtype
    if inp.dtype in (torch.float16, torch.bfloat16):
        inp = inp.to(torch.float32)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out_index = radix_sort_indices(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out_index = torch.movedim(out_index, -1, dim)
    return out_index