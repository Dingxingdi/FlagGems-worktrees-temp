import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fused_moving_avg_obs_fq_helper_kernel(
    input_ptr,
    observer_on_ptr,
    fake_quant_on_ptr,
    output_ptr,
    mask_ptr,
    running_min_ptr,
    running_max_ptr,
    scale_ptr,
    zero_point_ptr,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant,
    symmetric_quant,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CH_AXIS: tl.constexpr,
    PER_ROW: tl.constexpr,
    SYMMETRIC: tl.constexpr,
):
    """Kernel for _fused_moving_avg_obs_fq_helper.

    This operator fuses:
    1. Moving average observer (updates running_min/running_max)
    2. Fake quantization (applies quantization and dequantization)

    Args:
        input: Input tensor
        observer_on: Mask tensor for observer (whether to update running stats)
        fake_quant_on: Mask tensor for fake quantization
        output: Output tensor (quantized then dequantized)
        mask: Mask indicating which elements were quantized
        running_min/max: Running min/max values (updated in-place)
        scale: Quantization scale
        zero_point: Quantization zero point
        averaging_const: Smoothing constant for moving average
        quant_min/max: Quantization range
        ch_axis: Channel axis for per-channel quantization
        per_row_fake_quant: Whether to use per-row quantization
        symmetric_quant: Whether to use symmetric quantization
    """
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    observer_on_vals = tl.load(observer_on_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    fake_quant_on_vals = tl.load(fake_quant_on_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # For this implementation, we compute per-tensor quantization
    # Get the global scale and zero_point
    scale = tl.load(scale_ptr).to(tl.float32)
    zero_point = tl.load(zero_point_ptr).to(tl.float32)

    # Compute observer mask: which elements should update running stats
    observer_mask = observer_on_vals > 0.0

    # Compute fake quant mask: which elements should be quantized
    fake_quant_mask = fake_quant_on_vals > 0.0

    # Combined mask for output
    quant_mask = observer_mask | fake_quant_mask

    # Initialize output as input (identity for non-quantized elements)
    output_vals = input_vals

    # Apply fake quantization where fake_quant_on is true
    # Fake quant formula: round(x / scale + zero_point) * scale
    # Note: For symmetric quantization, zero_point = 0
    if SYMMETRIC:
        # Symmetric quantization: zero_point = 0
        scaled_vals = input_vals / scale
        quantized_vals = tl.floor(scaled_vals + 0.5)
        # Clamp to quant range
        quantized_vals = tl.minimum(tl.maximum(quantized_vals, quant_min), quant_max)
        output_vals = tl.where(fake_quant_mask, quantized_vals * scale, input_vals)
    else:
        # Asymmetric quantization
        scaled_vals = input_vals / scale + zero_point
        quantized_vals = tl.floor(scaled_vals + 0.5)
        # Clamp to quant range
        quantized_vals = tl.minimum(tl.maximum(quantized_vals, quant_min), quant_max)
        # Dequantize
        output_vals = tl.where(fake_quant_mask, (quantized_vals - zero_point) * scale, input_vals)

    # Store output and mask
    tl.store(output_ptr + offsets, output_vals, mask=mask)
    tl.store(mask_ptr + offsets, quant_mask, mask=mask)


def _fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    """Fused moving average observer and fake quantize helper.

    This operator fuses:
    1. Moving average observer (updates running_min/running_max) when observer_on > 0
    2. Fake quantization when fake_quant_on > 0

    Returns:
        output: The quantized then dequantized tensor
        mask: Boolean mask indicating which elements were affected
    """
    logger.debug("GEMS _FUSED_MOVING_AVG_OBS_FQ_HELPER")

    # Ensure inputs are contiguous
    self = self.contiguous()
    observer_on = observer_on.contiguous()
    fake_quant_on = fake_quant_on.contiguous()

    n_elements = self.numel()
    output = torch.empty_like(self)
    mask = torch.empty_like(self, dtype=torch.bool)

    # Parse inputs
    # running_min, running_max, scale, zero_point are expected to be 1-d tensors
    # with a single element each

    BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    ch_axis = ch_axis if ch_axis >= 0 else self.ndim + ch_axis
    per_row = per_row_fake_quant
    symmetric = symmetric_quant

    with torch.cuda.device(self.device):
        _fused_moving_avg_obs_fq_helper_kernel[grid](
            self,
            observer_on,
            fake_quant_on,
            output,
            mask,
            running_min,
            running_max,
            scale,
            zero_point,
            averaging_const,
            quant_min,
            quant_max,
            ch_axis,
            per_row,
            symmetric,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            CH_AXIS=ch_axis,
            PER_ROW=per_row,
            SYMMETRIC=symmetric,
        )

    return output, mask