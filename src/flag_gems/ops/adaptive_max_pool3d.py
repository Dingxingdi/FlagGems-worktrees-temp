import logging

import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@triton.jit
def adaptive_max_pool3d_kernel(
    # Pointers
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Input shapes
    in_n,
    in_c,
    in_d,
    in_h,
    in_w,
    # Output shapes
    out_d,
    out_h,
    out_w,
):
    """Forward kernel for adaptive max pooling 3D.

    For output position (d, h, w), the input region is:
    - d_start = (d * in_d) // out_d
    - d_end = ((d + 1) * in_d + out_d - 1) // out_d  # ceil division
    Similar for h and w.
    """
    # Get position
    pid = tle.program_id(0)

    # Total output elements per batch-channel
    total_out = out_d * out_h * out_w

    # Compute batch and channel indices
    # pid = batch_idx * in_c * total_out + c_idx * total_out + out_idx
    batch_idx = pid // (in_c * total_out)
    remaining = pid % (in_c * total_out)
    c_idx = remaining // total_out
    out_idx = remaining % total_out

    # Compute d, h, w from output index
    d_out = out_idx // (out_h * out_w)
    h_out = (out_idx // out_w) % out_h
    w_out = out_idx % out_w

    # Compute the input region for this output using PyTorch's formula
    # start = (i * in_size) // out_size
    # end = ((i + 1) * in_size + out_size - 1) // out_size  (ceil division)
    d_start = (d_out * in_d) // out_d
    d_end = ((d_out + 1) * in_d + out_d - 1) // out_d
    d_end = tl.minimum(d_end, in_d)

    h_start = (h_out * in_h) // out_h
    h_end = ((h_out + 1) * in_h + out_h - 1) // out_h
    h_end = tl.minimum(h_end, in_h)

    w_start = (w_out * in_w) // out_w
    w_end = ((w_out + 1) * in_w + out_w - 1) // out_w
    w_end = tl.minimum(w_end, in_w)

    dtype = input_ptr.type.element_ty
    min_value = get_dtype_min(dtype)
    max_val = min_value

    # Input base pointer for this (n, c)
    # Input has shape (in_n, in_c, in_d, in_h, in_w)
    input_base_ptr = input_ptr + batch_idx * in_stride_n + c_idx * in_stride_c

    # Find max in the input region
    for d_in in range(d_start, d_end):
        for h_in in range(h_start, h_end):
            for w_in in range(w_start, w_end):
                input_offset = d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                inp_val = tl.load(input_base_ptr + input_offset).to(tl.float32)
                max_val = tl.maximum(max_val, inp_val)

    # Store result
    # Output has shape (in_n, in_c, out_d, out_h, out_w)
    # batch 0, channel 0 -> output[0, 0, ...]
    # batch 0, channel 1 -> output[0, 1, ...]
    # batch 1, channel 0 -> output[1, 0, ...]
    # offset = batch_idx * in_c * out_d * out_h * out_w + c_idx * out_d * out_h * out_w + out_offset
    batch_offset = batch_idx * in_c * out_d * out_h * out_w
    c_offset = c_idx * out_d * out_h * out_w
    out_offset = d_out * out_h * out_w + h_out * out_w + w_out
    out_base_ptr = output_ptr + batch_offset + c_offset
    tl.store(out_base_ptr + out_offset, max_val.to(dtype))


def adaptive_max_pool3d(
    input: torch.Tensor,
    output_size: tuple,
    return_indices: bool = False,
):
    """Adaptive max pooling 3D.

    Args:
        input: Input tensor of shape (N, C, D, H, W)
        output_size: Target output size (single int or 3-tuple)
        return_indices: Whether to return pooling indices

    Returns:
        output: Output tensor of shape (N, C, out_D, out_H, out_W)
    """
    logger.debug("GEMS ADAPTIVE_MAX_POOL3D FORWARD")

    # Handle output_size
    if isinstance(output_size, int):
        out_d = out_h = out_w = output_size
    elif isinstance(output_size, (tuple, list)):
        if len(output_size) == 1:
            out_d = out_h = out_w = output_size[0]
        elif len(output_size) == 3:
            out_d, out_h, out_w = output_size
        else:
            raise ValueError(f"output_size must be int or 3-tuple, got {len(output_size)}-tuple")
    else:
        raise TypeError(f"output_size must be int or tuple, got {type(output_size)}")

    # Ensure contiguous input
    input = input.contiguous()

    in_n, in_c, in_d, in_h, in_w = input.shape

    # Allocate output tensor
    output = torch.empty((in_n, in_c, out_d, out_h, out_w), device=input.device, dtype=input.dtype)

    if output.numel() == 0:
        return output

    # Launch kernel - one thread per output element
    grid = lambda META: (in_n * in_c * out_d * out_h * out_w,)

    adaptive_max_pool3d_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.stride(4),
        in_n,
        in_c,
        in_d,
        in_h,
        in_w,
        out_d,
        out_h,
        out_w,
    )

    return output


def adaptive_max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor = None,
):
    """Backward pass for adaptive max pooling 3D.

    This is a placeholder that returns zeros.
    """
    logger.debug("GEMS ADAPTIVE_MAX_POOL3D BACKWARD")

    # Return zeros with same shape as input
    grad_input = torch.zeros_like(input)
    return grad_input