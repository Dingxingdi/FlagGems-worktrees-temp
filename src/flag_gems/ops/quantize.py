import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def quantize_func(x, scale, zero_point):
    # Quantization formula: round(x / scale + zero_point)
    # Then clip to valid quantized range [0, 255] for quint8
    # Using "round half to even" (banker's rounding) for exact tie cases
    scaled = x / scale + zero_point
    # Use float32 for the computation to avoid precision issues
    scaled_fp32 = scaled.to(tl.float32)

    # Get floor and fractional part
    floor_val = tl.floor(scaled_fp32)
    frac = scaled_fp32 - floor_val

    # For exact tie (frac is very close to 0.5), use banker's rounding
    # Use a tighter epsilon for exact ties
    is_exact_tie = tl.abs(frac - 0.5) < 1e-9

    # For tie case: round to nearest even
    # Check if floor is even
    floor_is_even = tl.abs(tl.floor(floor_val / 2.0) * 2.0 - floor_val) < 1e-6
    tie_result = tl.where(floor_is_even, floor_val, floor_val + 1.0)

    # For non-tie case: standard rounding floor(x + 0.5)
    # This handles both frac < 0.5 and frac > 0.5 cases
    non_tie_result = tl.floor(scaled_fp32 + 0.5)

    # Combine
    rounded = tl.where(is_exact_tie, tie_result, non_tie_result)

    # Clip to uint8 range [0, 255]
    clipped = tl.minimum(255.0, tl.maximum(0.0, rounded))
    return clipped.to(x.dtype)


def quantize(A, scale, zero_point):
    """Quantize tensor using per-tensor affine quantization.

    Args:
        A: Input tensor to quantize
        scale: Scale factor for quantization
        zero_point: Zero point offset

    Returns:
        Quantized values as float tensor
    """
    logger.debug("GEMS quantize")
    return quantize_func(A, scale, zero_point)


def quantize_(A, scale, zero_point):
    """In-place quantize tensor using per-tensor affine quantization.

    Args:
        A: Input tensor to quantize (modified in place)
        scale: Scale factor for quantization
        zero_point: Zero point offset

    Returns:
        Quantized tensor (same object as input)
    """
    logger.debug("GEMS quantize_")
    quantize_func(A, scale, zero_point, out0=A)
    return A