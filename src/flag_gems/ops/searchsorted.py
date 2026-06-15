import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@triton.jit
def searchsorted_kernel_impl(
    sorted_sequence_ptr: tl.tensor,
    values_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,  # number of rows (prod of broadcasted outer dims)
    N: int,  # size of innermost dimension of sorted_sequence
    K: int,  # innermost dimension of values
    sorted_seq_stride: int,  # stride for moving to next element in a row
    right: tl.constexpr,
):
    """
    Binary search kernel for searchsorted.

    Each thread handles one value and searches in the corresponding sorted_sequence row.

    Args:
        sorted_sequence_ptr: pointer to sorted_sequence data, shape (M, N)
        values_ptr: pointer to values data, shape (M, K)
        out_ptr: pointer to output data, shape (M, K)
        M: number of rows
        N: size of innermost dimension of sorted_sequence
        K: innermost dimension of values
        sorted_seq_stride: stride between consecutive elements in a row of sorted_sequence
        right: if True, find upper bound; if False, find lower bound
    """
    pid = tle.program_id(0)
    num_ctas = tle.num_programs(0)

    # Each thread handles one value at position (row_idx, col_idx)
    row_idx = pid // K
    col_idx = pid % K

    while row_idx < M:
        # Load the value to search for
        values_offset = row_idx * K + col_idx
        value = tl.load(values_ptr + values_offset)

        # Binary search in sorted_sequence[row_idx, :]
        low = 0
        high = N
        # Maximum iterations needed: ceil(log2(N))
        # For N up to 65536, we need at most 16 iterations
        for _ in range(16):
            if low >= high:
                break
            mid = (low + high) // 2
            sorted_offset = row_idx * N + mid
            mid_val = tl.load(sorted_sequence_ptr + sorted_offset * sorted_seq_stride)

            if right:
                # upper bound: find first index where sorted_sequence[i] > value
                cond = mid_val <= value
            else:
                # lower bound: find first index where sorted_sequence[i] >= value
                cond = mid_val < value

            low = tl.where(cond, mid + 1, low)
            high = tl.where(cond, high, mid)

        # Store the result
        out_offset = row_idx * K + col_idx
        tl.store(out_ptr + out_offset, low)

        # Move to next row (grid-stride loop)
        row_idx += num_ctas


@libentry()
@triton.jit
def searchsorted_kernel(
    sorted_sequence_ptr: tl.tensor,
    values_ptr: tl.tensor,
    out_ptr: tl.tensor,
    M: int,
    N: int,
    K: int,
    sorted_seq_stride: int,
    right: tl.constexpr,
    tiles_per_cta: int,
):
    pid = tle.program_id(0)
    ctas_num = tle.num_programs(0)

    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        searchsorted_kernel_impl(
            global_pid,
            sorted_sequence_ptr,
            values_ptr,
            out_ptr,
            M,
            N,
            K,
            sorted_seq_stride,
            right,
        )


def _compute_broadcast_shape(seq_outer, values_outer):
    """Compute the broadcasted shape of two shapes."""
    if not seq_outer:
        return values_outer if values_outer else []
    if not values_outer:
        return seq_outer

    # Pad the shorter shape with 1s
    len_diff = len(seq_outer) - len(values_outer)
    if len_diff > 0:
        values_outer = [1] * len_diff + list(values_outer)
    elif len_diff < 0:
        seq_outer = [1] * (-len_diff) + list(seq_outer)

    # Broadcast each dimension
    result = []
    for s, v in zip(seq_outer, values_outer):
        if s == v:
            result.append(s)
        elif s == 1:
            result.append(v)
        elif v == 1:
            result.append(s)
        else:
            raise ValueError(f"Cannot broadcast shapes {tuple(seq_outer)} and {tuple(values_outer)}")
    return result


def searchsorted(sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None, out=None):
    """
    Find indices where values should be inserted to maintain order in sorted_sequence.

    This is a binary search implementation for the searchsorted operator.

    Args:
        sorted_sequence: N-D tensor, sorted on innermost dimension
        values: N-D tensor or scalar
        out_int32: if True, output dtype is torch.int32; else torch.int64
        right: if True, return upper bound; if False, return lower bound
        side: "left" or "right", preferred over right parameter
        sorter: optional tensor of indices that sort sorted_sequence
        out: optional output tensor

    Returns:
        Tensor with same shape as values containing insertion indices
    """
    logger.debug("GEMS SEARCHSORTED")

    # Handle side parameter (preferred over right)
    if side is not None:
        if side == "left":
            right = False
        elif side == "right":
            right = True
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side}")

    if side is not None and right:
        raise ValueError("Cannot set side='left' and right=True simultaneously")

    # Handle scalar values
    is_scalar = not torch.is_tensor(values)
    if is_scalar:
        values = torch.tensor(values, device=sorted_sequence.device, dtype=sorted_sequence.dtype)

    # If sorter is provided, we need to unsort sorted_sequence first
    if sorter is not None:
        sorted_sequence = torch.gather(sorter, -1, sorted_sequence.to(torch.int64)).to(sorted_sequence.dtype)

    sorted_sequence = sorted_sequence.contiguous()
    values = values.contiguous()

    # Get shapes
    seq_shape = sorted_sequence.shape  # (..., N)
    values_shape = values.shape  # (..., K)

    seq_ndim = sorted_sequence.dim()
    values_ndim = values.dim()

    # The innermost dimension of sorted_sequence is the search space (size N)
    N = seq_shape[-1]
    seq_outer = seq_shape[:-1]  # (M_dims,) or ()

    if values_ndim == 0:
        # Scalar values - broadcasts to all rows
        values_outer = ()
        values_inner = 1
    else:
        values_outer = values_shape[:-1]  # (...,) or ()
        values_inner = values_shape[-1]  # K

    # Compute broadcasted outer dimensions
    try:
        broadcast_outer = _compute_broadcast_shape(list(seq_outer), list(values_outer))
    except ValueError as e:
        # Fall back to torch's broadcasting
        broadcast_outer = []
        # This shouldn't happen if PyTorch can broadcast, but just in case

    M = 1
    for dim_size in broadcast_outer:
        M *= dim_size
    K = values_inner

    # Reshape for kernel - both tensors become (M, N) and (M, K)
    sorted_seq_2d = sorted_sequence.reshape(M, N)
    values_2d = values.reshape(M, K)

    out_dtype = torch.int32 if out_int32 else torch.int64

    if out is None:
        out = torch.empty_like(values_2d, dtype=out_dtype)
    else:
        out = out.reshape(M, K)

    # Grid configuration - one thread per output element
    ctas_num = min(65536, triton.cdiv(M * K, 256))
    tiles_per_cta = triton.cdiv(M * K, ctas_num * 256)

    grid = (ctas_num,)

    with torch_device_fn.device(sorted_sequence.device.index):
        searchsorted_kernel[grid](
            sorted_seq_2d,
            values_2d,
            out,
            M,
            N,
            K,
            sorted_seq_2d.stride(0),
            right,
            tiles_per_cta=tiles_per_cta,
            num_warps=8,
        )

    # Reshape output to match values shape
    if is_scalar:
        # Return a scalar tensor (0-d)
        return out.squeeze().item() if M == 1 and K == 1 else out.squeeze()
    else:
        # Compute output shape: broadcast_outer + (K,)
        output_shape = tuple(broadcast_outer) + (K,) if broadcast_outer else (K,)
        return out.reshape(output_shape)