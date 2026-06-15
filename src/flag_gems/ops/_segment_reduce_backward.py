import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _segment_reduce_backward_sum_kernel(
    grad_ptr,
    out_ptr,
    lengths_ptr,
    n,
    segment_size,
    stride_grad,
    stride_out,
    stride_lengths,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward kernel for sum reduction."""
    pid = tle.program_id(0)
    segment_offset = pid * segment_size

    # Load the length of this segment
    segment_length = tl.load(lengths_ptr + pid)

    # Each thread handles one element in the segment
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < segment_length

    # Load gradient for this segment
    grad_val = tl.load(grad_ptr + pid).to(tl.float32)

    # For sum: gradient is scattered equally to each position
    # Each element in the segment gets the same gradient
    out_val = tl.where(mask, grad_val, 0.0)

    # Store the result
    out_ptr_row = out_ptr + segment_offset
    tl.store(out_ptr_row + offsets, out_val.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _segment_reduce_backward_mean_kernel(
    grad_ptr,
    out_ptr,
    lengths_ptr,
    n,
    segment_size,
    stride_grad,
    stride_out,
    stride_lengths,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward kernel for mean reduction."""
    pid = tle.program_id(0)

    # Load the length of this segment
    segment_length = tl.load(lengths_ptr + pid).to(tl.float32)
    # Avoid division by zero
    segment_length = tl.where(segment_length == 0.0, 1.0, segment_length)

    segment_offset = pid * segment_size

    # Each thread handles one element in the segment
    offsets = tl.arange(0, BLOCK_SIZE)
    # Use original length for mask (as int)
    seg_len_int = tl.load(lengths_ptr + pid)
    mask = offsets < seg_len_int

    # Load gradient for this segment
    grad_val = tl.load(grad_ptr + pid).to(tl.float32)

    # For mean: gradient is divided by segment length
    out_val = tl.where(mask, grad_val / segment_length, 0.0)

    # Store the result
    out_ptr_row = out_ptr + segment_offset
    tl.store(out_ptr_row + offsets, out_val.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _segment_reduce_backward_max_kernel(
    grad_ptr,
    output_ptr,
    data_ptr,
    out_ptr,
    lengths_ptr,
    n,
    segment_size,
    stride_grad,
    stride_data,
    stride_out,
    stride_lengths,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward kernel for max reduction."""
    pid = tle.program_id(0)

    segment_offset = pid * segment_size
    segment_length = tl.load(lengths_ptr + pid)

    # Each thread handles one element in the segment
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < segment_length

    # Load gradient for this segment
    grad_val = tl.load(grad_ptr + pid).to(tl.float32)

    # Load output (max value) for this segment
    output_val = tl.load(output_ptr + pid).to(tl.float32)

    # Load data values for this segment
    data_offsets = segment_offset + offsets
    data_vals = tl.load(data_ptr + data_offsets, mask=mask, other=0.0).to(tl.float32)

    # For max: gradient goes only to positions where data equals output
    # Load as float32 for precise comparison
    data_vals_fp32 = tl.load(data_ptr + data_offsets, mask=mask, other=0.0)
    output_val_fp32 = tl.load(output_ptr + pid)
    is_max = data_vals_fp32 == output_val_fp32
    out_val = tl.where(mask & is_max, grad_val, 0.0)

    # Store the result
    out_ptr_row = out_ptr + segment_offset
    tl.store(out_ptr_row + offsets, out_val.to(out_ptr.dtype.element_ty), mask=mask)


def _segment_reduce_backward(grad, output, data, reduce, lengths=None, offsets=None, axis=0, initial=None):
    """Compute gradient of segment_reduce backward pass."""
    logger.debug("GEMS _SEGMENT_REDUCE_BACKWARD")

    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported for segment_reduce_backward")

    if lengths is None and offsets is None:
        raise ValueError("Either lengths or offsets must be provided")

    if lengths is not None and offsets is not None:
        raise ValueError("Only one of lengths or offsets can be provided")

    grad = grad.contiguous()
    output = output.contiguous()
    data = data.contiguous()

    # Determine number of segments
    if lengths is not None:
        n = lengths.numel()
        lengths = lengths.contiguous()
    else:
        n = offsets.numel() - 1
        offsets = offsets.contiguous()

    # Determine segment size (max length)
    if lengths is not None:
        segment_size = lengths.max().item()
    else:
        segment_size = (offsets[1:] - offsets[:-1]).max().item()

    # Allocate output
    out = torch.zeros_like(data)

    BLOCK_SIZE = min(triton.next_power_of_2(segment_size), 1024)

    with torch_device_fn.device(grad.device):
        if reduce == "sum":
            _segment_reduce_backward_sum_kernel[(n,)](
                grad,
                out,
                lengths if lengths is not None else offsets,
                n,
                segment_size,
                grad.stride(0),
                out.stride(0),
                (lengths if lengths is not None else offsets).stride(0),
                BLOCK_SIZE,
            )
        elif reduce == "mean":
            _segment_reduce_backward_mean_kernel[(n,)](
                grad,
                out,
                lengths if lengths is not None else offsets,
                n,
                segment_size,
                grad.stride(0),
                out.stride(0),
                (lengths if lengths is not None else offsets).stride(0),
                BLOCK_SIZE,
            )
        elif reduce == "max":
            _segment_reduce_backward_max_kernel[(n,)](
                grad,
                output,
                data,
                out,
                lengths if lengths is not None else offsets,
                n,
                segment_size,
                grad.stride(0),
                data.stride(0),
                out.stride(0),
                (lengths if lengths is not None else offsets).stride(0),
                BLOCK_SIZE,
            )
        else:
            raise NotImplementedError(f"Reduce type '{reduce}' is not supported")

    return out