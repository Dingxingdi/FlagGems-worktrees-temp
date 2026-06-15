import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def upsample_bilinear2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    align_corners_int: tl.constexpr,
    scale_h,
    scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward pass for bilinear upsampling.

    For each output pixel, we compute which input pixels it receives contributions from
    and distribute the gradient using bilinear weights.
    """
    # Each program handles a subset of output pixels
    pid = tl.program_id(0)
    num_els = N * C * OH * OW

    # Calculate the starting offset for this program
    start_offs = pid * BLOCK_SIZE
    offs = start_offs + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_els

    # Load gradients for this block and convert to float32 for accurate accumulation
    grad = tl.load(grad_output_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute N, C, H, W indices for each element
    # Flatten: (N, C, OH, OW) -> (N*C*OH*OW)
    nC = N * C
    ow = offs % OW
    oh = (offs // OW) % OH
    c = (offs // (OH * OW)) % C
    n = offs // (nC * OH * OW)

    # Compute the source position in the input tensor
    # For bilinear interpolation:
    # With align_corners=False: t = (x + 0.5) / scale - 0.5
    # With align_corners=True: t = x * (IH-1) / (OH-1) if OH > 1 else 0

    # Compute t_h and t_w (continuous input coordinates)
    if align_corners_int:
        # align_corners=True
        if OH > 1:
            t_h = oh * (IH - 1) / (OH - 1)
        else:
            t_h = tl.zeros(oh.shape, dtype=tl.float32)
        if OW > 1:
            t_w = ow * (IW - 1) / (OW - 1)
        else:
            t_w = tl.zeros(ow.shape, dtype=tl.float32)
    else:
        # align_corners=False (default)
        # Map output position to input position: t = (x + 0.5) / scale - 0.5
        # where scale = OH/IH (forward upsampling scale)
        t_h = (oh.to(tl.float32) + 0.5) / scale_h - 0.5
        t_w = (ow.to(tl.float32) + 0.5) / scale_w - 0.5

    # Get floor and ceil indices
    ih0 = tl.floor(t_h).to(tl.int32)
    iw0 = tl.floor(t_w).to(tl.int32)
    ih1 = ih0 + 1
    iw1 = iw0 + 1

    # Clamp to valid input range
    ih0_clamped = tl.minimum(ih0, IH - 1)
    ih1_clamped = tl.minimum(ih1, IH - 1)
    iw0_clamped = tl.minimum(iw0, IW - 1)
    iw1_clamped = tl.minimum(iw1, IW - 1)

    # Ensure non-negative
    ih0_valid = tl.maximum(ih0_clamped, 0)
    ih1_valid = tl.maximum(ih1_clamped, 0)
    iw0_valid = tl.maximum(iw0_clamped, 0)
    iw1_valid = tl.maximum(iw1_clamped, 0)

    # Compute fractional parts (bilinear weights)
    h1_frac = t_h - tl.floor(t_h).to(tl.float32)
    w1_frac = t_w - tl.floor(t_w).to(tl.float32)
    h0_frac = 1.0 - h1_frac
    w0_frac = 1.0 - w1_frac

    # Compute weights for the 4 corners
    weight_00 = h0_frac * w0_frac  # top-left
    weight_01 = h0_frac * w1_frac  # top-right
    weight_10 = h1_frac * w0_frac  # bottom-left
    weight_11 = h1_frac * w1_frac  # bottom-right

    # Compute output stride for writing to grad_input
    # grad_input layout: (N, C, IH, IW)
    ih_stride = IW
    iw_stride = 1

    # Compute base offset for each (n, c)
    base_offset = n * C * IH * IW + c * IH * IW

    # Compute offsets for the 4 corners in grad_input
    # corner (ih0, iw0) - top-left
    offset_00 = base_offset + ih0_valid * ih_stride + iw0_valid * iw_stride
    # corner (ih0, iw1) - top-right
    offset_01 = base_offset + ih0_valid * ih_stride + iw1_valid * iw_stride
    # corner (ih1, iw0) - bottom-left
    offset_10 = base_offset + ih1_valid * ih_stride + iw0_valid * iw_stride
    # corner (ih1, iw1) - bottom-right
    offset_11 = base_offset + ih1_valid * ih_stride + iw1_valid * iw_stride

    # Scatter the gradients using atomic_add
    # Each thread writes to 4 different locations
    grad_00 = grad * weight_00
    grad_01 = grad * weight_01
    grad_10 = grad * weight_10
    grad_11 = grad * weight_11

    tl.atomic_add(grad_input_ptr + offset_00, grad_00, mask=mask)
    tl.atomic_add(grad_input_ptr + offset_01, grad_01, mask=mask)
    tl.atomic_add(grad_input_ptr + offset_10, grad_10, mask=mask)
    tl.atomic_add(grad_input_ptr + offset_11, grad_11, mask=mask)


def upsample_bilinear2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    """
    Backward function for upsample_bilinear2d.

    Args:
        grad_output: Gradient with respect to the output (N, C, OH, OW)
        output_size: Output size (OH, OW)
        input_size: Input size (N, C, IH, IW)
        align_corners: Whether to align corners
        scales_h: Scale factor for height
        scales_w: Scale factor for width

    Returns:
        Gradient with respect to the input (N, C, IH, IW)
    """
    logger.debug("GEMS upsample_bilinear2d_backward")
    assert grad_output.device.type == "cuda"
    assert grad_output.ndim == 4, "The ndim of grad_output must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"
    assert len(input_size) == 4, "The len of input_size must be 4"

    N, C, IH, IW = input_size
    OH, OW = output_size

    # Compute scale factors (forward scale: output/input)
    # For backward, we use these to map output coords to input coords
    if scales_h is not None:
        scale_h = scales_h  # This is the forward scale (OH/IH)
    else:
        scale_h = OH / IH if IH > 0 else 1.0

    if scales_w is not None:
        scale_w = scales_w  # This is the forward scale (OW/IW)
    else:
        scale_w = OW / IW if IW > 0 else 1.0

    # Always allocate in float32 for accurate accumulation
    grad_input_fp32 = torch.zeros(
        (N, C, IH, IW), device=grad_output.device, dtype=torch.float32
    )

    # Calculate grid
    total_elements = N * C * OH * OW
    BLOCK_SIZE = 256
    num_programs = triton.cdiv(total_elements, BLOCK_SIZE)

    grid = (num_programs,)

    upsample_bilinear2d_backward_kernel[grid](
        grad_output,
        grad_input_fp32,
        N,
        C,
        OH,
        OW,
        IH,
        IW,
        1 if align_corners else 0,
        scale_h,
        scale_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Convert back to original dtype
    grad_input = grad_input_fp32.to(grad_output.dtype)

    return grad_input