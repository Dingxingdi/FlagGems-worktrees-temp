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
def segment_reduce_sum_kernel(
    data_ptr,
    seg_starts_ptr,
    seg_lengths_ptr,
    output_ptr,
    num_segments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid >= num_segments:
        return

    seg_start = tl.load(seg_starts_ptr + pid)
    seg_length = tl.load(seg_lengths_ptr + pid)
    seg_end = seg_start + seg_length

    # Reduce over the segment
    sum_val = tl.cast(0.0, tl.float32)
    for idx in range(seg_start, seg_end):
        val = tl.load(data_ptr + idx)
        sum_val = sum_val + tl.cast(val, tl.float32)

    tl.store(output_ptr + pid, sum_val)


@libentry()
@triton.jit
def segment_reduce_max_kernel(
    data_ptr,
    seg_starts_ptr,
    seg_lengths_ptr,
    output_ptr,
    num_segments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid >= num_segments:
        return

    seg_start = tl.load(seg_starts_ptr + pid)
    seg_length = tl.load(seg_lengths_ptr + pid)
    seg_end = seg_start + seg_length

    # Initialize with very negative value
    max_val = tl.cast(-float('inf'), tl.float32)
    for idx in range(seg_start, seg_end):
        val = tl.load(data_ptr + idx)
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))

    tl.store(output_ptr + pid, max_val)


@libentry()
@triton.jit
def segment_reduce_min_kernel(
    data_ptr,
    seg_starts_ptr,
    seg_lengths_ptr,
    output_ptr,
    num_segments: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid >= num_segments:
        return

    seg_start = tl.load(seg_starts_ptr + pid)
    seg_length = tl.load(seg_lengths_ptr + pid)
    seg_end = seg_start + seg_length

    # Initialize with very positive value
    min_val = tl.cast(float('inf'), tl.float32)
    for idx in range(seg_start, seg_end):
        val = tl.load(data_ptr + idx)
        min_val = tl.minimum(min_val, tl.cast(val, tl.float32))

    tl.store(output_ptr + pid, min_val)


def segment_reduce(data, reduce, *, lengths=None, indices=None, offsets=None,
                   axis=0, unsafe=False, initial=None):
    """
    Performs segment reduction operation.

    Args:
        data: Input tensor
        reduce: Reduction method ("sum", "mean", "max", "min", "prod")
        lengths: Optional tensor specifying segment lengths
        indices: Optional tensor specifying segment indices
        offsets: Optional tensor specifying segment offsets
        axis: Axis along which to perform reduction (only 0 supported for now)
        unsafe: Whether to skip bounds checking
        initial: Initial value for reduction
    """
    logger.debug("GEMS SEGMENT_REDUCE")

    if axis != 0:
        raise NotImplementedError("segment_reduce only supports axis=0 for now")

    if indices is not None:
        raise NotImplementedError("segment_reduce with indices is not supported yet")

    if data.ndim != 1:
        raise NotImplementedError("segment_reduce only supports 1D tensors for now")

    # Determine segment information
    if lengths is not None and offsets is None:
        lengths_tensor = lengths.to(data.device).to(torch.int64)
        num_segments = lengths_tensor.numel()
    elif offsets is not None and lengths is None:
        offsets_cpu = offsets.to('cpu').to(torch.int64)
        num_segments = offsets_cpu.numel() - 1
        # Convert offsets to lengths (on CPU)
        lengths_tensor = torch.zeros(num_segments, dtype=torch.int64, device='cpu')
        for i in range(num_segments):
            lengths_tensor[i] = offsets_cpu[i + 1] - offsets_cpu[i]
    else:
        raise ValueError("segment_reduce requires either lengths or offsets, not both")

    # Precompute segment start positions on CPU, then copy to GPU
    # If using offsets, seg_starts are simply the first num_segments elements
    seg_starts = torch.zeros(num_segments, dtype=torch.int64, device='cpu')
    if offsets is not None:
        # Use offsets directly as segment starts
        seg_starts = offsets_cpu[:num_segments].clone()
    else:
        # Compute cumulative sum of lengths
        for i in range(num_segments):
            seg_starts[i] = lengths_tensor[:i].sum().item() if i > 0 else 0

    # Create GPU tensors
    seg_starts_gpu = seg_starts.to(data.device)
    lengths_gpu = lengths_tensor.to(data.device)

    # Handle empty segments case
    if data.numel() == 0:
        output = torch.empty((num_segments,), dtype=data.dtype, device=data.device)
        if initial is not None:
            output.fill_(initial)
        else:
            if reduce == "sum":
                output.fill_(0)
            elif reduce == "max":
                output.fill_(float('-inf'))
            elif reduce == "min":
                output.fill_(float('inf'))
            elif reduce == "prod":
                output.fill_(1)
            elif reduce == "mean":
                output.fill_(float('nan'))
        return output

    # Create output tensor
    output = torch.empty((num_segments,), dtype=torch.float32, device=data.device)

    with torch_device_fn.device(data.device):
        BLOCK_SIZE = 128

        if reduce == "sum":
            segment_reduce_sum_kernel[(num_segments,)](
                data, seg_starts_gpu, lengths_gpu, output, num_segments, BLOCK_SIZE
            )
        elif reduce == "max":
            segment_reduce_max_kernel[(num_segments,)](
                data, seg_starts_gpu, lengths_gpu, output, num_segments, BLOCK_SIZE
            )
        elif reduce == "min":
            segment_reduce_min_kernel[(num_segments,)](
                data, seg_starts_gpu, lengths_gpu, output, num_segments, BLOCK_SIZE
            )
        else:
            raise NotImplementedError(f"segment_reduce with reduce={reduce} not supported yet")

    # Convert output to original dtype
    output = output.to(data.dtype)

    return output