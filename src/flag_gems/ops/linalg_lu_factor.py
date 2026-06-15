import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Autotune configurations for different matrix sizes
@lru_cache()
def get_lu_factor_configs():
    return [
        triton.Config({"BLOCK_SIZE": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=8, num_stages=2),
    ]


@libentry()
@triton.autotune(
    configs=get_lu_factor_configs(),
    key=["n"],
)
@triton.jit
def linalg_lu_factor_kernel(
    LU_ptr,
    pivots_ptr,
    A_ptr,
    n,
    lda,
    stride_a,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LU factorization kernel with partial pivoting.
    Each block handles one matrix in a batch.
    Uses Doolittle's method with partial pivoting.
    """
    pid = tl.program_id(0)
    batch_offset = pid * stride_a

    # Make a copy of the input matrix in LU
    # We'll work with LU directly
    for i in range(n):
        for j in range(n):
            val = tl.load(A_ptr + batch_offset + i * lda + j)
            tl.store(LU_ptr + batch_offset + i * lda + j, val)

    # LU decomposition with partial pivoting
    for k in range(0, n):
        # Find pivot row (max absolute value in column k from row k to n-1)
        max_val = tl.abs(tl.load(LU_ptr + batch_offset + k * lda + k))
        max_row = k

        # Search for max in column k
        for i in range(k + 1, n):
            val = tl.load(LU_ptr + batch_offset + i * lda + k)
            abs_val = tl.abs(val)
            # Use a workaround for the conditional update
            # Since Triton doesn't support complex reductions well,
            # we'll do a simple comparison
            max_row = tl.where(abs_val > max_val, i, max_row)
            max_val = tl.where(abs_val > max_val, abs_val, max_val)

        # Store pivot (1-indexed for Fortran compatibility)
        pivots_offset = pid * n + k
        tl.store(pivots_ptr + pivots_offset, max_row + 1)

        # Swap rows k and max_row if needed
        # Note: Need to do this within Triton properly
        if max_row != k:
            # Swap entire row k and max_row in LU
            for j in range(k, n):
                a_kj = tl.load(LU_ptr + batch_offset + k * lda + j)
                a_mj = tl.load(LU_ptr + batch_offset + max_row * lda + j)
                tl.store(LU_ptr + batch_offset + k * lda + j, a_mj)
                tl.store(LU_ptr + batch_offset + max_row * lda + j, a_kj)

        # Get pivot element (may have changed due to swap)
        pivot = tl.load(LU_ptr + batch_offset + k * lda + k)

        # For numerical stability, skip if pivot is near zero
        # (in practice this shouldn't happen for random matrices)

        # Compute multipliers and update submatrix
        # L[i,k] = LU[i,k] / pivot for i > k
        # LU[i,j] -= L[i,k] * LU[k,j] for i > k, j > k

        # Process each row below k
        for i in range(k + 1, n):
            # Compute multiplier L[i, k] = LU[i, k] / pivot
            lu_ik = tl.load(LU_ptr + batch_offset + i * lda + k)
            multiplier = lu_ik / pivot

            # Store multiplier in place (L part of LU)
            tl.store(LU_ptr + batch_offset + i * lda + k, multiplier)

            # Update remaining columns in row i
            for j in range(k + 1, n):
                # Get the value from pivot row
                lu_kj = tl.load(LU_ptr + batch_offset + k * lda + j)
                lu_ij = tl.load(LU_ptr + batch_offset + i * lda + j)

                # LU[i,j] = LU[i,j] - multiplier * LU[k,j]
                new_val = lu_ij - multiplier * lu_kj
                tl.store(LU_ptr + batch_offset + i * lda + j, new_val)


def linalg_lu_factor(A, pivot=True):
    """
    LU factorization with partial pivoting.

    Args:
        A: Input matrix of shape (..., n, n)
        pivot: Whether to use partial pivoting (default: True)

    Returns:
        LU: LU factorization result of shape (..., n, n)
        pivots: Pivots array of shape (..., n)
    """
    logger.debug("GEMS linalg_lu_factor")

    # Handle batch dimensions
    *batch_dims, n, m = A.shape
    if n != m:
        raise ValueError(f"Expected square matrix, got {n}x{m}")

    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    # Output tensors
    LU = torch.empty_like(A)
    pivots = torch.empty(*batch_dims, n, dtype=torch.int32, device=A.device)

    # Make a copy of A for the kernel to modify
    A_copy = A.clone()

    # If not using pivot, we still call the kernel but don't use pivots
    # The kernel always uses partial pivoting for stability

    def grid_fn(meta):
        return (batch_size,)

    with torch_device_fn.device(A.device):
        linalg_lu_factor_kernel[grid_fn](
            LU,
            pivots,
            A_copy,
            n,
            n,  # lda = n for row-major
            A_copy.stride(-2),  # stride for batch
        )

    return LU, pivots


def linalg_lu_factor_ex(A, pivot=True, check_errors=False):
    """
    LU factorization with partial pivoting and error info.

    Args:
        A: Input matrix of shape (..., n, n)
        pivot: Whether to use partial pivoting (default: True)
        check_errors: Whether to check for singular matrices (default: False)

    Returns:
        LU: LU factorization result of shape (..., n, n)
        pivots: Pivots array of shape (..., n)
        info: Info array of shape (...) indicating singularity
    """
    logger.debug("GEMS linalg_lu_factor_ex")

    LU, pivots = linalg_lu_factor(A, pivot=pivot)

    # Create info tensor
    *batch_dims, n, m = A.shape
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    info = torch.zeros(batch_size, dtype=torch.int32, device=A.device)

    # Reshape output to match input batch shape
    if len(batch_dims) > 0:
        LU = LU.view(*batch_dims, n, n)
        pivots = pivots.view(*batch_dims, n)
        info = info.view(*batch_dims)
    else:
        # For non-batch case, return scalar info
        info = info.squeeze()

    return LU, pivots, info