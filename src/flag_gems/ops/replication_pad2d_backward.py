import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def replication_pad2d_backward_kernel(
    grad_output_ptr,
    input_ptr,
    output_ptr,
    N,
    C,
    H,
    W,
    OW,
    OH,
    PAD_LEFT,
    PAD_RIGHT,
    PAD_TOP,
    PAD_BOTTOM,
    TOTAL_ELEMS,
    stride_grad_out_n,
    stride_grad_out_c,
    stride_grad_out_h,
    stride_grad_out_w,
    stride_in_n,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for replication_pad2d_backward.

    This kernel parallelizes over output (grad_input) positions.
    For each output position (n, c, ih, iw), we need to sum up gradients from
    all output (padded) positions that map to it.

    Forward mapping (output_padded -> input):
        ih_in = clamp(oh - PAD_TOP, 0, H - 1)
        iw_in = clamp(ow - PAD_LEFT, 0, W - 1)

    Backward: for each input position (ih, iw), sum gradients from all
    padded positions (oh, ow) where:
        ih = clamp(oh - PAD_TOP, 0, H - 1)
        iw = clamp(ow - PAD_LEFT, 0, W - 1)

    Equivalent to: oh in [ih + PAD_TOP - min(PAD_TOP, ih), ih + PAD_TOP + min(PAD_BOTTOM, H-1-ih)]
                   ow in [iw + PAD_LEFT - min(PAD_LEFT, iw), iw + PAD_LEFT + min(PAD_RIGHT, W-1-iw)]

    We compute this by iterating over the possible oh, ow range for each ih, iw.
    Since PAD values are small constants, we can unroll the loops.
    """
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL_ELEMS

    # Convert to flattened index
    offs64 = offs.to(tl.int64)

    N_i64 = tl.full([1], N, dtype=tl.int64)
    C_i64 = tl.full([1], C, dtype=tl.int64)
    H_i64 = tl.full([1], H, dtype=tl.int64)
    W_i64 = tl.full([1], W, dtype=tl.int64)
    OW_i64 = tl.full([1], OW, dtype=tl.int64)
    OH_i64 = tl.full([1], OH, dtype=tl.int64)
    PAD_LEFT_i64 = tl.full([1], PAD_LEFT, dtype=tl.int64)
    PAD_RIGHT_i64 = tl.full([1], PAD_RIGHT, dtype=tl.int64)
    PAD_TOP_i64 = tl.full([1], PAD_TOP, dtype=tl.int64)
    PAD_BOTTOM_i64 = tl.full([1], PAD_BOTTOM, dtype=tl.int64)

    # Unflatten to input indices (n, c, ih, iw)
    out_w = offs64 % W_i64
    tmp = offs64 // W_i64
    out_h = tmp % H_i64
    tmp = tmp // H_i64
    c = tmp % C_i64
    n = tmp // C_i64

    zero_i64 = tl.full([1], 0, dtype=tl.int64)
    Hm1_i64 = H_i64 - 1
    Wm1_i64 = W_i64 - 1

    # Compute the valid range of oh and ow that map to this input position
    # oh_start = ih + PAD_TOP - min(PAD_TOP, ih)
    # oh_end = ih + PAD_TOP + min(PAD_BOTTOM, H-1-ih)
    oh_start = out_h + PAD_TOP_i64 - tl.minimum(out_h, PAD_TOP_i64)
    oh_end = out_h + PAD_TOP_i64 + tl.minimum(PAD_BOTTOM_i64, Hm1_i64 - out_h)
    ow_start = out_w + PAD_LEFT_i64 - tl.minimum(out_w, PAD_LEFT_i64)
    ow_end = out_w + PAD_LEFT_i64 + tl.minimum(PAD_RIGHT_i64, Wm1_i64 - out_w)

    # Clamp to valid range
    OHm1_i64 = OH_i64 - 1
    OWm1_i64 = OW_i64 - 1
    oh_start = tl.maximum(zero_i64, tl.minimum(oh_start, OHm1_i64))
    oh_end = tl.maximum(zero_i64, tl.minimum(oh_end, OHm1_i64))
    ow_start = tl.maximum(zero_i64, tl.minimum(ow_start, OWm1_i64))
    ow_end = tl.maximum(zero_i64, tl.minimum(ow_end, OWm1_i64))

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Since padding values are typically small (1-4), we can unroll the loops
    # The maximum range is PAD_TOP + PAD_BOTTOM + 1 which is at most 9 for typical paddings
    # We'll handle different padding sizes with separate code paths

    # Actually, let's use a different approach: iterate over all possible
    # (oh - oh_start) and (ow - ow_start) offsets, where the offsets are limited
    # We can do this with a fixed maximum number of iterations

    # For typical padding values (1-4), the max range is small enough to unroll
    # Let's compute the range size and iterate

    # Since Triton doesn't support dynamic loops well, let's use a fixed iteration count
    # based on maximum possible padding (assuming padding <= 4 for typical use cases)
    # We'll just iterate with fixed upper bound and mask invalid iterations

    # Get the range sizes - clamp to max 9 to handle typical cases
    oh_range = oh_end - oh_start + 1
    ow_range = ow_end - ow_start + 1

    # Use fixed max iterations (assuming max padding of 4 in each direction)
    MAX_PAD = 9  # (4 + 4 + 1)

    for i in range(MAX_PAD):
        oh_offset = i // 3
        ow_offset = i % 3
        oh = oh_start + oh_offset
        ow = ow_start + ow_offset

        # Check if within computed range using arithmetic instead of tl.any
        oh_valid = (oh >= oh_start) & (oh <= oh_end)
        ow_valid = (ow >= ow_start) & (ow <= ow_end)
        # Combine using multiplication (works as logical AND for 0/1 values)
        valid = oh_valid * ow_valid

        # Load and accumulate using the valid mask for loading
        g = tl.load(grad_output_ptr + ((n * C_i64 + c) * OH_i64 + oh) * OW_i64 + ow, mask=mask & valid, other=0.0).to(tl.float32)
        acc = acc + g

    # Store result
    out_idx = ((n * C_i64 + c) * H_i64 + out_h) * W_i64 + out_w
    tl.store(output_ptr + out_idx, acc.to(output_ptr.dtype.element_ty), mask=mask)


def replication_pad2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    padding: tuple,
):
    """Compute the gradient of replication_pad2d.

    Args:
        grad_output: Gradient from the padded tensor, shape (N, C, H + pad_top + pad_bottom, W + pad_left + pad_right)
        input: Original input tensor, shape (N, C, H, W)
        padding: Tuple of (pad_left, pad_right, pad_top, pad_bottom)

    Returns:
        Gradient with respect to input, shape (N, C, H, W)
    """
    logger.debug("GEMS REPLICATION_PAD2D_BACKWARD")

    pad_left, pad_right, pad_top, pad_bottom = padding

    # Validate input
    assert input.dim() in (3, 4), f"Expected 3D or 4D input, got {input.dim()}D"
    assert input.is_cuda, "Input must be CUDA tensor"
    assert grad_output.is_cuda, "grad_output must be CUDA tensor"
    assert grad_output.device == input.device, "Tensors must be on same device"

    # Handle 3D (C, H, W) vs 4D (N, C, H, W)
    if input.dim() == 3:
        input = input.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)

    N, C, H, W = input.shape
    OH, OW = grad_output.shape[-2:]

    assert OH == H + pad_top + pad_bottom, f"Expected OH={H + pad_top + pad_bottom}, got {OH}"
    assert OW == W + pad_left + pad_right, f"Expected OW={W + pad_left + pad_right}, got {OW}"

    # Ensure contiguous
    grad_output = grad_output.contiguous()
    input = input.contiguous()

    # Allocate output
    grad_input = torch.empty_like(input)

    if grad_input.numel() == 0:
        return grad_input.squeeze(0) if input.dim() == 3 else grad_input

    BLOCK_SIZE = 1024
    total_elems = grad_input.numel()
    grid = (triton.cdiv(total_elems, BLOCK_SIZE),)

    replication_pad2d_backward_kernel[grid](
        grad_output,
        input,
        grad_input,
        N,
        C,
        H,
        W,
        OW,
        OH,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        total_elems,
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_input.squeeze(0) if input.dim() == 3 else grad_input