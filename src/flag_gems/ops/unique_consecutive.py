import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def unique_consecutive_flat_ne_impl(
    global_pid,
    input_ptr: tl.tensor,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    global_ctas_num: int,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks

    # load input
    a = tl.load(input_ptr + i0, mask=mask)

    # load previous element
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(input_ptr + i0_prev, mask=mask)

    # compute ne_result: 1 if (i == 0 or x[i] != x[i-1]), 0 otherwise
    ne_result = tl.where(i0 > 0, a != b, 1)
    tl.store(ne_result_ptr + i0, ne_result, mask=mask)

    # compute tile_sum
    tile_sum = tl.sum(ne_result)
    tile_sum_mask = global_pid < global_ctas_num
    tl.store(tile_sum_ptr + global_pid, tile_sum, mask=tile_sum_mask)


@libentry()
@triton.jit
def unique_consecutive_flat_ne_kernel(
    input_ptr: tl.tensor,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    global_ctas_num: int,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        unique_consecutive_flat_ne_impl(
            global_pid,
            input_ptr,
            ne_result_ptr,
            tile_sum_ptr,
            global_ctas_num,
            num_tasks,
            tile_size,
        )


@triton.jit
def unique_consecutive_flat_scatter_impl(
    global_pid,
    total,
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    input_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    ctas_num: tl.constexpr,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tile_size: tl.constexpr,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
):
    offset = global_pid * tile_size
    r = tl.arange(0, tile_size)
    i0 = offset + r
    mask = i0 < num_tasks

    # load input
    input_vals = tl.load(input_ptr + i0, mask=mask)

    # load tile_sum
    p = tl.arange(0, next_power_global_ctas_num)
    pre_tile_sum_mask = (
        (p >= global_pid - ctas_num)
        & (p < global_pid)
        & (p >= 0)
        & (p < global_ctas_num)
    )
    pre_tile_sum = tl.load(tile_sum_ptr + p, mask=pre_tile_sum_mask, other=0)

    # cumsum
    total += tl.sum(pre_tile_sum)
    ne_result = tl.load(ne_result_ptr + i0, mask=mask)
    ne_result_i1 = ne_result.to(tl.int1)
    ne_result_i32 = ne_result.to(tl.int32)
    cumsum = tl.cumsum(ne_result_i32)
    # cumsum is 1-indexed, subtract 1 to get 0-indexed positions
    cumsum = cumsum - 1
    cumsum += total

    # tile_sum - for each block, compute the sum of ne_result
    # This gives us the count of unique elements in this block
    block_sum = tl.sum(ne_result_i32)
    tl.store(tile_sum_ptr + global_pid, block_sum, mask=global_pid < global_ctas_num)

    # data_out: scatter_(to=cumsum, input_vals)
    tl.store(data_out_ptr + cumsum, input_vals, mask=mask)

    # inverse_indices: for original index i, store where it went in output
    if return_inverse:
        tl.store(inverse_indices_ptr + i0, cumsum, mask=mask)

    # idx for counts
    if return_counts:
        idx_mask = ((i0 == 0) | ne_result_i1) & mask
        # Store the original indices where new unique elements start
        tl.store(tile_sum_ptr + global_ctas_num + cumsum, i0, mask=idx_mask)

    return total


@libentry()
@triton.jit
def unique_consecutive_flat_scatter_kernel(
    ne_result_ptr: tl.tensor,
    tile_sum_ptr: tl.tensor,
    input_ptr: tl.tensor,
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    ctas_num: int,
    global_ctas_num: int,
    next_power_global_ctas_num: tl.constexpr,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    if one_tile_per_cta:  # monolitic kernel style
        unique_consecutive_flat_scatter_impl(
            pid,
            0,
            ne_result_ptr,
            tile_sum_ptr,
            input_ptr,
            data_out_ptr,
            inverse_indices_ptr,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tile_size,
            return_inverse,
            return_counts,
        )
    else:  # grid-stride-loop style kernel
        total = tl.zeros([1], dtype=tl.int64)
        for j in range(0, tiles_per_cta):
            global_pid = pid + j * ctas_num
            total = unique_consecutive_flat_scatter_impl(
                global_pid,
                total,
                ne_result_ptr,
                tile_sum_ptr,
                input_ptr,
                data_out_ptr,
                inverse_indices_ptr,
                ctas_num,
                global_ctas_num,
                next_power_global_ctas_num,
                num_tasks,
                tile_size,
                return_inverse,
                return_counts,
            )


@triton.jit
def output_counts_flat_impl(
    global_pid,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,
    counts_ptr: tl.tensor,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)

    # load idx
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    idx = tl.load(idx_ptr + i0, mask=mask)

    # load idx_next
    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)

    # diff: counts[i] = idx[i+1] - idx[i]
    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)

    # store counts
    tl.store(counts_ptr + i0, counts, mask=mask)


@libentry()
@triton.jit
def output_counts_flat_kernel(
    idx_ptr: tl.tensor,
    origin_num_tasks: int,
    counts_ptr: tl.tensor,
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        output_counts_flat_impl(
            global_pid,
            idx_ptr,
            origin_num_tasks,
            counts_ptr,
            num_tasks,
            tile_size,
        )


def unique_consecutive_flat(
    input: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    num_tasks = input.numel()
    next_power_num_tasks = triton.next_power_of_2(num_tasks)
    tile_size = min(8192, next_power_num_tasks)
    global_ctas_num = triton.cdiv(num_tasks, tile_size)
    if global_ctas_num <= 8192:
        min_tile_size = 512 if global_ctas_num > 32 else 256
        tile_size = max(
            min_tile_size,
            min(triton.next_power_of_2(global_ctas_num), next_power_num_tasks),
        )
        global_ctas_num = triton.cdiv(num_tasks, tile_size)
    next_power_global_ctas_num = triton.next_power_of_2(global_ctas_num)
    ctas_num = global_ctas_num if global_ctas_num < 32768 else 8192
    tiles_per_cta = triton.cdiv(num_tasks, tile_size * ctas_num)
    num_warps = 8 if tiles_per_cta == 1 else 32
    grid = (ctas_num, 1, 1)

    # allocate tensor
    ne_result = torch.empty_like(input, dtype=torch.int32)
    # tile_sum needs extra space for storing indices when return_counts is True
    tile_sum_extra = 1 if return_counts else 0
    tile_sum = torch.empty(
        (global_ctas_num + tile_sum_extra,), dtype=torch.int64, device=input.device
    )
    data_out = torch.empty_like(input)
    inverse_indices = None
    if return_inverse:
        inverse_indices = torch.empty_like(input, dtype=torch.int64)

    # launch kernel
    with torch_device_fn.device(input.device.index):
        unique_consecutive_flat_ne_kernel[grid](
            input,
            ne_result,
            tile_sum,
            global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            num_warps=num_warps,
        )
        unique_consecutive_flat_scatter_kernel[grid](
            ne_result,
            tile_sum,
            input,
            data_out,
            inverse_indices,
            ctas_num,
            global_ctas_num,
            next_power_global_ctas_num,
            num_tasks,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            one_tile_per_cta=tiles_per_cta == 1,
            return_inverse=return_inverse,
            return_counts=return_counts,
            num_warps=num_warps,
        )

        # tile_sum now contains the sum of ne_result for each block
        # Compute the total using cumsum on CPU
        out_size = torch.sum(tile_sum[:global_ctas_num]).item()
        counts = None

        if return_counts:
            # Use the last element of tile_sum as the idx buffer
            idx = tile_sum[global_ctas_num : global_ctas_num + out_size]
            counts = torch.empty_like(idx)
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,
                counts,
                out_size,
                tiles_per_cta,
                tile_size,
                num_warps=num_warps,
            )

    return data_out[:out_size], inverse_indices, counts


def _unique_consecutive(
    in0: torch.Tensor,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int = None,
):
    logger.debug("GEMS unique_consecutive")

    if dim is not None:
        # Handle dim parameter
        dim = dim % in0.ndim
        if dim != 0:
            # Move the dimension to the front
            in0 = in0.transpose(0, dim)

        # Process along dim 0
        original_shape = in0.shape
        in0 = in0.contiguous()
        in0_flat = in0.view(-1)

        # Use unique_consecutive on flattened data
        out_flat, inverse_flat, counts_flat = unique_consecutive_flat(
            in0_flat, return_inverse, return_counts
        )

        # Compute output shape
        first_dim_size = out_flat.shape[0]
        out_shape = (first_dim_size,) + original_shape[1:]
        out = out_flat.view(out_shape)

        # Handle inverse_indices
        inverse_indices = None
        if return_inverse:
            inverse_indices = inverse_flat.view(original_shape)

        # Handle counts
        counts = None
        if return_counts:
            counts = counts_flat

        # Move dimension back if needed
        if dim != 0:
            out = out.transpose(0, dim)
            if inverse_indices is not None:
                inverse_indices = inverse_indices.transpose(0, dim)

        return out, inverse_indices, counts

    else:
        # No dim specified, operate on flattened tensor
        in0_flat = in0.contiguous().ravel()
        out, inverse_indices, counts = unique_consecutive_flat(
            in0_flat, return_inverse, return_counts
        )

        return out, inverse_indices, counts