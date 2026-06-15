import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=1),
    ],
    key=["N"],
)
@triton.jit
def cholesky_kernel(
    A,
    L,
    N,
):
    """
    Compute Cholesky decomposition A = L @ L^T (lower triangular form).
    Single program computes the entire matrix sequentially.
    """
    # Only one program runs - compute all rows sequentially
    for row_idx in range(0, N):
        # Compute elements in this row: L[row_idx, col_idx] for col_idx <= row_idx
        for col_idx in range(0, row_idx + 1):
            # Load A[row_idx, col_idx]
            a_elem = tl.load(A + row_idx * N + col_idx)

            if row_idx == col_idx:
                # Diagonal element: L[i,i] = sqrt(A[i,i] - sum(L[i,k]^2 for k < i))
                sum_squares = 0.0
                for k in range(0, col_idx):
                    L_kk = tl.load(L + row_idx * N + k)
                    sum_squares = sum_squares + L_kk * L_kk

                result = tl.sqrt(a_elem - sum_squares)
            else:
                # Off-diagonal: L[row_idx, col_idx] = (A[row_idx, col_idx] - sum(L[row_idx,k] * L[col_idx,k] for k < col_idx)) / L[col_idx, col_idx]
                sum_prod = 0.0
                for k in range(0, col_idx):
                    L_row_k = tl.load(L + row_idx * N + k)
                    L_col_k = tl.load(L + col_idx * N + k)
                    sum_prod = sum_prod + L_row_k * L_col_k

                # Load the diagonal element L[col_idx, col_idx]
                L_diag = tl.load(L + col_idx * N + col_idx)
                result = (a_elem - sum_prod) / L_diag

            # Store the result
            tl.store(L + row_idx * N + col_idx, result)


def cholesky(A, upper=False):
    """
    Compute the Cholesky decomposition of a symmetric positive-definite matrix.

    Args:
        A: Input tensor of shape (*, n, n) - batch of symmetric positive-definite matrices
        upper: If True, return upper triangular matrix U such that A = U^T @ U
               If False (default), return lower triangular matrix L such that A = L @ L^T

    Returns:
        L or U: Tensor of same shape as input, lower or upper triangular matrix
    """
    logger.debug("GEMS cholesky")

    # Handle batch dimensions
    original_shape = A.shape
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("Input matrix must be square")
    if A.shape[-1] == 0:
        return A.clone()

    # Flatten batch dimensions for processing
    batch_dims = original_shape[:-2]
    N = original_shape[-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    # For numerical stability, we work in float32 for float16/bfloat16
    dtype = A.dtype
    compute_dtype = dtype
    if dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    # Reshape to (batch, n, n) for processing
    A_flat = A.reshape(batch_size, N, N).to(compute_dtype) if dtype != compute_dtype else A.reshape(batch_size, N, N)

    # Ensure matrix is symmetric positive definite: A_sym = (A + A^T) / 2
    A_sym = (A_flat + A_flat.transpose(-2, -1)) / 2

    # Create output tensor
    L = torch.zeros_like(A_sym, dtype=compute_dtype)

    # Process each matrix in the batch
    for b in range(batch_size):
        A_input = A_sym[b]
        L_output = L[b]

        # Launch kernel - single program computes entire matrix
        grid = lambda meta: (1,)
        cholesky_kernel[grid](A_input, L_output, N)

    # Reshape output back to original shape
    result = L.reshape(original_shape)

    # Transpose if upper=True to get U such that A = U^T @ U
    if upper:
        result = result.transpose(-2, -1)

    # Convert back to original dtype if needed
    if dtype != compute_dtype:
        result = result.to(dtype)

    return result


def cholesky_(A, upper=False):
    """
    In-place version of cholesky.
    """
    logger.debug("GEMS cholesky_")
    result = cholesky(A, upper=upper)
    A.copy_(result)
    return A