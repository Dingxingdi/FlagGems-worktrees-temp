import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn

device = device.name
logger = logging.getLogger(__name__)


@triton.jit
def upsample_nearest2d_backward_kernel(
    ptr_grad_input,
    ptr_grad_output,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Flatten N*C into a single dimension for simpler indexing
    # Each thread handles one position in the output
    pid = tl.program_id(0).to(tl.int64)

    # Total number of output elements
    numel_output = N * C * OH * OW
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel_output

    # Unflatten idx to get n, c, oh, ow
    # idx = ((n * C) + c) * OH * OW + oh * OW + ow
    tmp = idx // (OH * OW)
    nc = tmp % C
    n = tmp // C
    oh_ow = idx % (OH * OW)
    oh = oh_ow // OW
    ow = oh_ow % OW

    # Compute corresponding input position
    ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    # Compute offsets
    offset_output = idx
    # Input offset: ((n * C) + c) * IH * IW + ih * IW + iw
    offset_input = ((n * C + nc) * IH + ih) * IW + iw

    # Load from grad_output and atomic add to grad_input
    # Convert to float32 for atomic add to avoid bfloat16 issues
    grad = tl.load(ptr_grad_output + offset_output, mask=mask, other=0.0).to(tl.float32)
    # Use masked atomic add - only add when mask is True
    tl.atomic_add(ptr_grad_input + offset_input, grad, mask=mask)


def upsample_nearest2d_backward(
    grad_output: torch.Tensor,
    output_size: Tuple[int, int],
    input_size: Tuple[int, int, int, int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE NEAREST2D BACKWARD")
    assert grad_output.device.type == device
    assert len(input_size) == 4, "The len of input_size must be 4 (N, C, H, W)"
    assert len(output_size) == 2, "The len of output_size must be 2 (H, W)"

    OH, OW = output_size
    N, C, IH, IW = input_size

    # Validate grad_output shape matches output_size
    assert grad_output.shape == (N, C, OH, OW), (
        f"grad_output shape {grad_output.shape} does not match "
        f"expected shape ({N}, {C}, {OH}, {OW})"
    )

    # Compute scale ratio from input/output sizes
    # The output_size already incorporates any scaling, so we derive the ratio
    reciprocal_scale_h = IH / OH if OH > 0 else 0.0
    reciprocal_scale_w = IW / OW if OW > 0 else 0.0

    # Flatten grad_output to match the kernel's 1D indexing
    grad_output_flat = grad_output.view(-1)

    # Use float32 for accumulation to support bfloat16 and float16
    # Then convert back to original dtype
    grad_input = torch.zeros((N, C, IH, IW), device=grad_output.device, dtype=torch.float32)

    # Select BLOCK_SIZE based on total threads
    total_threads = N * C * OH * OW
    if total_threads <= 256:
        BLOCK_SIZE = 256
    elif total_threads <= 512:
        BLOCK_SIZE = 512
    elif total_threads <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048

    grid = (triton.cdiv(total_threads, BLOCK_SIZE),)

    with torch_device_fn.device(grad_output.device):
        upsample_nearest2d_backward_kernel[grid](
            grad_input,
            grad_output_flat,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            reciprocal_scale_h,
            reciprocal_scale_w,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return grad_input.to(grad_output.dtype)