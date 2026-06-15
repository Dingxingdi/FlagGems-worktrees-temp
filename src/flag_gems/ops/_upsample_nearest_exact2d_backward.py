import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _upsample_nearest_exact2d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    sN_grad_out,
    sC_grad_out,
    sH_grad_out,
    sW_grad_out,
    sN_grad_in,
    sC_grad_in,
    sH_grad_in,
    sW_grad_in,
    use_scales: tl.constexpr,
    scale_h,
    scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    nc_stride = tl.num_programs(axis=1)
    NC = N * C
    nc_iter = tl.program_id(axis=1)

    # Compute n and c from flattened plane index
    n = nc_iter // C
    c = nc_iter - n * C

    # Each thread handles BLOCK_SIZE output pixels
    base_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow = base_idx % OW
    oh = base_idx // OW % OH

    mask = base_idx < (OH * OW)

    # Compute source indices for each output pixel
    # Using the nearest-exact formula: ih = floor((oh + 0.5) * IH / OH)
    if use_scales:
        # With explicit scales: ih = min(floor(oh / scale_h), IH - 1)
        ih_f = oh.to(tl.float32) / scale_h
        iw_f = ow.to(tl.float32) / scale_w
        ih = tl.minimum(tl.floor(ih_f).to(tl.int32), IH - 1)
        iw = tl.minimum(tl.floor(iw_f).to(tl.int32), IW - 1)
    else:
        # Without scales: ih = min(floor((oh + 0.5) * IH / OH), IH - 1)
        ih_f = (oh.to(tl.float32) + 0.5) * (IH / OH)
        iw_f = (ow.to(tl.float32) + 0.5) * (IW / OW)
        ih = tl.minimum(tl.floor(ih_f).to(tl.int32), IH - 1)
        iw = tl.minimum(tl.floor(iw_f).to(tl.int32), IW - 1)

    # Compute offsets
    grad_out_offset = n * sN_grad_out + c * sC_grad_out + oh * sH_grad_out + ow * sW_grad_out
    grad_in_offset = n * sN_grad_in + c * sC_grad_in + ih * sH_grad_in + iw * sW_grad_in

    # Load gradient as float32 and atomic add to input (as float32)
    grad_val = tl.load(grad_out_ptr + grad_out_offset, mask=mask).to(tl.float32)

    # Use atomic add to accumulate gradients from multiple output pixels to same input pixel
    tl.atomic_add(grad_in_ptr + grad_in_offset, grad_val, mask=mask)


def _upsample_nearest_exact2d_backward(
    grad_output,
    output_size,
    input_size,
    scales_h=None,
    scales_w=None,
):
    logger.debug("GEMS _UPSAMPLE_NEAREST_EXACT2D_BACKWARD")

    OH, OW = output_size
    N, C, IH, IW = input_size

    if not grad_output.is_cuda or not grad_output.device.type == "cuda":
        return torch.ops.aten._upsample_nearest_exact2d_backward(
            grad_output, output_size, input_size, scales_h, scales_w
        )

    # Allocate output gradient tensor (always float32 for accumulation)
    grad_input = torch.zeros((N, C, IH, IW), dtype=torch.float32, device=grad_output.device)

    if grad_input.numel() == 0:
        return grad_input

    sN_grad_out = grad_output.stride(0)
    sC_grad_out = grad_output.stride(1)
    sH_grad_out = grad_output.stride(2)
    sW_grad_out = grad_output.stride(3)
    sN_grad_in = grad_input.stride(0)
    sC_grad_in = grad_input.stride(1)
    sH_grad_in = grad_input.stride(2)
    sW_grad_in = grad_input.stride(3)

    # Determine if using scales or computing from sizes
    use_scales = scales_h is not None and scales_w is not None

    total_threads = OH * OW
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_threads, BLOCK_SIZE), N * C)

    with torch_device_fn.device(grad_output.device):
        _upsample_nearest_exact2d_backward_kernel[grid](
            grad_output,
            grad_input,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            sN_grad_out,
            sC_grad_out,
            sH_grad_out,
            sW_grad_out,
            sN_grad_in,
            sC_grad_in,
            sH_grad_in,
            sW_grad_in,
            use_scales=use_scales,
            scale_h=float(scales_h) if use_scales else 1.0,
            scale_w=float(scales_w) if use_scales else 1.0,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Convert back to the original dtype
    return grad_input.to(grad_output.dtype)