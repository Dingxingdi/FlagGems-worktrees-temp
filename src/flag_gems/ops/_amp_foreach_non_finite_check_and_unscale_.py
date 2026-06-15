import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _check_and_unscale_kernel(
    in_ptr,
    found_inf_ptr,
    inv_scale,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values as float32 for computation
    x = tl.load(in_ptr + offsets * 1, mask=mask, other=0.0).to(tl.float32)

    # Check for non-finite (inf or nan)
    # nan is not equal to itself, and inf fails the isfinite check
    is_nan = x != x
    is_inf = tl.abs(x) == float("inf")
    is_non_finite = is_nan | is_inf

    # If any non-finite found in this block, increment found_inf using atomic add
    # Note: This may increment found_inf multiple times if multiple tensors
    # have non-finite values, but we clip to 1.0 at the end
    if tl.sum(is_non_finite) > 0:
        tl.atomic_add(found_inf_ptr, 1.0)

    # Scale by inv_scale
    scaled = x * inv_scale.to(tl.float32)

    # Store back as original dtype
    tl.store(in_ptr + offsets * 1, scaled.to(in_ptr.dtype.element_ty), mask=mask)


def _amp_foreach_non_finite_check_and_unscale_(
    self: list, found_inf: torch.Tensor, inv_scale: torch.Tensor
):
    """Check for non-finite values and unscale tensors in-place.

    Args:
        self: List of tensors to check and unscale
        found_inf: Output tensor that will be set to 1.0 if any non-finite is found
        inv_scale: Inverse scale factor to multiply tensors with
    """
    logger.debug("GEMS _AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE_")

    # Reset found_inf to 0.0
    found_inf.fill_(0.0)

    # Get inv_scale value as a Python scalar
    inv_scale_val = inv_scale.item()

    for tensor in self:
        n_elements = tensor.numel()
        if n_elements == 0:
            continue

        # Ensure tensor is contiguous for 1D processing
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Calculate grid size
        BLOCK_SIZE = 1024
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

        # Launch kernel for this tensor
        grid = (num_blocks,)
        _check_and_unscale_kernel[grid](
            tensor,
            found_inf,
            inv_scale_val,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Clip found_inf to 1.0 - if any tensor had non-finite values,
    # found_inf will be >= 1.0, but AMP only cares if it's > 0
    if found_inf.item() > 1.0:
        found_inf.fill_(1.0)


def _amp_foreach_non_finite_check_and_unscale__list(
    self: list, found_inf: torch.Tensor, inv_scale: torch.Tensor
):
    """In-place version - same as functional version since tensors are modified in-place."""
    _amp_foreach_non_finite_check_and_unscale_(self, found_inf, inv_scale)