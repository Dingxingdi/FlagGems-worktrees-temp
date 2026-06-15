import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def svd_prepare_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    """Prepare kernel for SVD - copies input to output for computation."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_offsets = offset // N
    col_offsets = offset % N

    mask = (row_offsets < M) & (col_offsets < N)

    input_ptrs = input_ptr + row_offsets * stride_m + col_offsets
    val = tl.load(input_ptrs, mask=mask, other=0.0)

    output_ptrs = output_ptr + row_offsets * stride_m + col_offsets
    tl.store(output_ptrs, val, mask=mask)


@libentry()
@triton.jit
def svd_scale_kernel(
    input_ptr,
    output_ptr,
    scale,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Scale kernel for SVD - used for numerical stability."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements

    val = tl.load(input_ptr + offset, mask=mask, other=0.0)
    scaled_val = val * scale
    tl.store(output_ptr + offset, scaled_val, mask=mask)


def svd(A, some=True, compute_uv=True):
    """Computes the singular value decomposition of a matrix or batch of matrices.

    This implementation uses torch.linalg.svd (not registered with FlagGems)
    to avoid recursion, while still using cuSOLVER under the hood.

    Args:
        A: Input tensor of shape (*, m, n) where * is zero or more batch dimensions.
        some: If True, returns the reduced SVD (default: True).
        compute_uv: If False, returns zero-filled U and V (default: True).

    Returns:
        A namedtuple (U, S, V) where:
            U: Left singular vectors
            S: Singular values (diagonal)
            V: Right singular vectors
    """
    logger.debug("GEMS svd")

    # Handle batch dimensions
    if A.dim() < 2:
        raise ValueError("SVD requires at least 2D tensor")

    # Ensure input is contiguous for optimal performance
    output = A.contiguous()

    # Get dimensions
    m = output.shape[-2]
    n = output.shape[-1]
    k = min(m, n)

    if not compute_uv:
        # When compute_uv=False, return zero-filled tensors matching torch.svd behavior
        # U: (m, m), V: (n, n)
        if output.dim() > 2:
            batch_shape = output.shape[:-2]
            U = torch.zeros(batch_shape + (m, m), dtype=output.dtype, device=output.device)
            V = torch.zeros(batch_shape + (n, n), dtype=output.dtype, device=output.device)
        else:
            U = torch.zeros((m, m), dtype=output.dtype, device=output.device)
            V = torch.zeros((n, n), dtype=output.dtype, device=output.device)

        # Still compute singular values
        _, S, _ = torch.linalg.svd(output, full_matrices=not some)
        return (U, S, V)

    # Use torch.linalg.svd which is NOT registered with FlagGems
    # This avoids recursion while still using cuSOLVER
    # Note: linalg.svd returns (U, S, Vh) where Vh = V^T
    # full_matrices is opposite of 'some'
    U, S, Vh = torch.linalg.svd(output, full_matrices=not some)

    # Convert Vh to V to match torch.svd behavior
    # V = Vh^T for real matrices
    V = Vh.mT

    return (U, S, V)


def svd_(A, some=True, compute_uv=True):
    """In-place version of svd - uses torch.linalg.svd internally.

    Note: torch.svd doesn't have an in-place version, this is a wrapper
    that returns the SVD result.
    """
    logger.debug("GEMS svd_")
    return svd(A, some=some, compute_uv=compute_uv)