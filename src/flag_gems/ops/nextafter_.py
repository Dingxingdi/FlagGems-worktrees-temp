import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def nextafter_func(x, y):
    # Get the integer type with same bit width as the float type
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    int_dtype = tl.core.get_int_dtype(num_bits, False)

    # If x is NaN, return NaN (nextafter of NaN is NaN)
    # If y is NaN, return NaN (nextafter towards NaN is NaN)
    x_is_nan = x != x
    y_is_nan = y != y

    # Cast float to integer bits for manipulation
    x_bits = x.to(int_dtype, bitcast=True)

    # IEEE 754 float bit ordering:
    # - For positive floats: higher float value = higher unsigned int bits
    # - For negative floats: higher float value = LOWER unsigned int bits
    #
    # The sign bit mask to extract the MSB
    sign_bit_mask = 1 << (num_bits - 1)
    x_is_negative = (x_bits & sign_bit_mask) != 0

    # Determine direction: y > x means +1, y < x means -1
    # We need to add +1 or -1 to the bits to move in the direction of y
    #
    # For positive floats (sign bit = 0):
    #   - y > x: increment bits (higher float = higher bits)
    #   - y < x: decrement bits
    #
    # For negative floats (sign bit = 1):
    #   - y > x: decrement bits (higher float = LOWER bits because sign bit is set)
    #   - y < x: increment bits
    #
    # This can be unified as:
    # - If x_is_negative: negate the increment
    # - Because for negatives, bit ordering is opposite to float ordering
    y_gt_x = y > x
    y_lt_x = y < x
    direction = y_gt_x.to(int_dtype) - y_lt_x.to(int_dtype)

    # Negate direction when x is negative
    # direction is 1 when y > x, -1 when y < x
    # For negative x, we need the opposite
    signed_direction = tl.where(x_is_negative, -direction, direction)

    # Apply increment/decrement
    next_bits = x_bits + signed_direction

    # Convert back to float
    result = next_bits.to(x.dtype, bitcast=True)

    # If x is NaN or y is NaN, return NaN
    # If x == y, return x (handled correctly since x == y for non-NaN means no move needed)
    # Otherwise return the modified result
    return tl.where(x_is_nan | y_is_nan | (x == y), x, result)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def nextafter_func_template(x, y):
    return nextafter_func(x, y)


def nextafter(x, y):
    logger.debug("GEMS NEXTAFTER")
    return nextafter_func_template(x, y)


def nextafter_(x, y):
    logger.debug("GEMS NEXTAFTER_")
    nextafter_func_template(x, y, out0=x)
    return x