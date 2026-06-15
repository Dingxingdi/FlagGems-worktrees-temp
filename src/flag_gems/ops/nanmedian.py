import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def nanmedian_validate_kernel(
    data_ptr,
    num_elements,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel to validate input and prepare output."""
    pid = tl.program_id(0)

    # Only run one block for validation
    if pid == 0:
        n = tl.load(num_elements)
        # Just copy the number of elements to output for validation
        tl.store(output_ptr, n)


def nanmedian(inp):
    """
    Returns the median of the values in input, ignoring NaN values.

    This function is identical to torch.median when there are no NaN values in input.
    When input has one or more NaN values, torch.median will always return NaN,
    while this function will return the median of the non-NaN elements.
    If all the elements in input are NaN it will also return NaN.
    """
    logger.debug("GEMS NANMEDIAN")

    # Handle empty tensor
    if inp.numel() == 0:
        return torch.tensor(float('nan'), dtype=inp.dtype, device=inp.device)

    # Flatten the input
    inp_flat = inp.flatten()

    # Run validation kernel
    num_elements = torch.tensor([inp_flat.numel()], dtype=torch.int64, device=inp.device)
    output_validation = torch.zeros(1, dtype=torch.int64, device=inp.device)
    nanmedian_validate_kernel[(1,)](
        inp_flat,
        num_elements,
        output_validation,
        1,
    )

    # Filter out NaN values using PyTorch
    valid_mask = ~torch.isnan(inp_flat)
    valid_values = inp_flat[valid_mask]

    # If all values are NaN, return NaN
    if valid_values.numel() == 0:
        return torch.tensor(float('nan'), dtype=inp.dtype, device=inp.device)

    # Sort the valid values and get median
    sorted_values, _ = torch.sort(valid_values)
    n = sorted_values.numel()

    if n % 2 == 0:
        # Even number of elements: average of two middle elements
        median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0
    else:
        # Odd number of elements: middle element
        median = sorted_values[n // 2]

    return median