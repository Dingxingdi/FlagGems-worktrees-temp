import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def assert_async_kernel(
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that checks if any element is zero and triggers assertion if so."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=1.0)

    # Check if any value is zero
    # For floating point, we need to check if the value is exactly zero
    # For integer types, check if value equals zero
    is_zero = values == 0

    # Use tl.sum to check if any element is zero across the block
    # If sum > 0, there's at least one zero
    zero_count = tl.sum(is_zero.to(tl.int32))

    # Assert if any zero is found
    # Note: This will cause a CUDA error if any zero is found
    tl.device_assert(zero_count == 0, "Assertion failed: tensor contains zero value")


def _assert_async(A: torch.Tensor):
    """
    Asynchronously assert that the contents of tensor are nonzero.

    For CUDA tensors, this checks if any element is zero and triggers
    an assertion failure if so. The check is performed asynchronously.

    Args:
        A: a tensor to test for zero values
    """
    logger.debug("GEMS _ASSERT_ASYNC")
    n_elements = A.numel()

    # Define block size
    BLOCK_SIZE = 128
    # Calculate grid
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    assert_async_kernel[grid](
        A,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Return None (void)
    return None