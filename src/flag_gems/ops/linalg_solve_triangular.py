import logging
import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def linalg_solve_triangular_lower_kernel(
    A,  # input triangular matrix (*, n, n)
    B,  # right-hand side (*, n, k)
    X,  # output (*, n, k)
    n,  # matrix dimension
    k,  # number of columns in B
    batch_size,
    row,  # current row to solve
    upper: tl.constexpr,
    left: tl.constexpr,
    unitriangular: tl.constexpr,
    stride_batch_a,
    stride_batch_b,
    stride_batch_x,
    stride_row_a,
    stride_row_b,
    stride_row_x,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for solving L @ X = B where L is lower triangular.
    This kernel processes one row at a time sequentially.
    """
    # Get batch index
    pid_b = tle.program_id(2)
    # Each program handles BLOCK_SIZE_K columns
    col_offset = tle.program_id(0) * BLOCK_SIZE_K
    cols = tl.arange(0, BLOCK_SIZE_K)
    mask = col_offset + cols < k

    # Compute offset for this batch
    batch_offset = pid_b * stride_batch_a

    # Load B[row, col_offset:col_offset+BLOCK_SIZE_K]
    b_ptrs = B + batch_offset + row * stride_row_b + (col_offset + cols)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)

    # For lower triangular, compute X[row] = (B[row] - L[row,:row] @ X[:row]) / L[row,row]
    # Process elements j < row
    if row > 0:
        sum_vals = tl.zeros((BLOCK_SIZE_K,), dtype=b_vals.dtype)

        # Iterate over j from 0 to row-1
        for j in range(0, row):
            # Load L[row, j]
            a_ptr = A + batch_offset + row * stride_row_a + j
            a_val = tl.load(a_ptr)

            # Load X[j, col_offset:col_offset+BLOCK_SIZE_K]
            x_ptrs = X + batch_offset + j * stride_row_x + (col_offset + cols)
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

            sum_vals += a_val * x_vals

        b_vals = b_vals - sum_vals

    # Divide by diagonal element (unless unitriangular)
    if not unitriangular:
        diag_ptr = A + batch_offset + row * stride_row_a + row
        diag_val = tl.load(diag_ptr)
        b_vals = b_vals / diag_val

    # Store result
    x_ptrs = X + batch_offset + row * stride_row_x + (col_offset + cols)
    tl.store(x_ptrs, b_vals, mask=mask)


@libentry()
@triton.jit
def linalg_solve_triangular_upper_kernel(
    A,  # input triangular matrix (*, n, n)
    B,  # right-hand side (*, n, k)
    X,  # output (*, n, k)
    n,  # matrix dimension
    k,  # number of columns in B
    batch_size,
    row,  # current row to solve
    upper: tl.constexpr,
    left: tl.constexpr,
    unitriangular: tl.constexpr,
    stride_batch_a,
    stride_batch_b,
    stride_batch_x,
    stride_row_a,
    stride_row_b,
    stride_row_x,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for solving U @ X = B where U is upper triangular.
    This kernel processes one row at a time sequentially.
    """
    # Get batch index
    pid_b = tle.program_id(2)
    # Each program handles BLOCK_SIZE_K columns
    col_offset = tle.program_id(0) * BLOCK_SIZE_K
    cols = tl.arange(0, BLOCK_SIZE_K)
    mask = col_offset + cols < k

    # Compute offset for this batch
    batch_offset = pid_b * stride_batch_a

    # Load B[row, col_offset:col_offset+BLOCK_SIZE_K]
    b_ptrs = B + batch_offset + row * stride_row_b + (col_offset + cols)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)

    # For upper triangular, compute X[row] = (B[row] - U[row,row:] @ X[row:]) / U[row,row]
    # Process elements j > row
    if row < n - 1:
        sum_vals = tl.zeros((BLOCK_SIZE_K,), dtype=b_vals.dtype)

        # Iterate over j from row+1 to n-1
        for j in range(row + 1, n):
            # Load U[row, j]
            a_ptr = A + batch_offset + row * stride_row_a + j
            a_val = tl.load(a_ptr)

            # Load X[j, col_offset:col_offset+BLOCK_SIZE_K]
            x_ptrs = X + batch_offset + j * stride_row_x + (col_offset + cols)
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

            sum_vals += a_val * x_vals

        b_vals = b_vals - sum_vals

    # Divide by diagonal element (unless unitriangular)
    if not unitriangular:
        diag_ptr = A + batch_offset + row * stride_row_a + row
        diag_val = tl.load(diag_ptr)
        b_vals = b_vals / diag_val

    # Store result
    x_ptrs = X + batch_offset + row * stride_row_x + (col_offset + cols)
    tl.store(x_ptrs, b_vals, mask=mask)


def linalg_solve_triangular(A, B, *, upper, left=True, unitriangular=False):
    """
    Solve a triangular linear system.

    Solves AX = B (if left=True) or XA = B (if left=False)
    where A is a triangular matrix.

    Args:
        A: Tensor of shape (*, n, n) or (*, k, k) if left=False
        B: Tensor of shape (*, n, k) or (*, k, n) if left=False
        upper: Whether A is upper triangular (True) or lower triangular (False)
        left: Whether to solve AX = B (True) or XA = B (False)
        unitriangular: If True, diagonal elements are assumed to be 1

    Returns:
        X: Solution tensor with same shape as B
    """
    logger.debug("GEMS linalg_solve_triangular")

    # Get matrix dimensions
    # For left=True: solves AX = B, A is (*, n, n), B is (*, n, k), X is (*, n, k)
    # For left=False: solves XA = B, A is (*, k, k), B is (*, n, k), X is (*, n, k)
    if left:
        n = A.shape[-1]
        k = B.shape[-1]
    else:
        n = B.shape[-2]  # Rows of B
        k = A.shape[-1]  # Size of A (k×k)

    # Handle broadcasting for batch dimensions
    batch_shape = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # Reshape for batch processing
    if left:
        A = A.expand(*batch_shape, n, n).contiguous()
    else:
        A = A.expand(*batch_shape, k, k).contiguous()
    B = B.expand(*batch_shape, n, k).contiguous()

    # Allocate output
    X = torch.empty_like(B)

    # Copy B to X initially (will be overwritten row by row)
    X.copy_(B)

    # Use the natural strides from the tensors
    # After expand+contiguous:
    # - If batch_size == 1: shape is (n, n), stride(0) = n (row stride)
    # - If batch_size > 1: shape is (batch_size, n, n), stride(0) = n*n (batch stride)
    if batch_size > 1:
        stride_batch_a = A.stride(0)
        stride_batch_b = B.stride(0)
        stride_batch_x = X.stride(0)
        stride_row_a = A.stride(1)
        stride_row_b = B.stride(1)
        stride_row_x = X.stride(1)
    else:
        # No batch dimension
        stride_batch_a = 0
        stride_batch_b = 0
        stride_batch_x = 0
        stride_row_a = A.stride(0)
        stride_row_b = B.stride(0)
        stride_row_x = X.stride(0)

    BLOCK_SIZE_K = 16

    if left:
        if upper:
            # Upper triangular, solving AX = B
            # Process rows from bottom to top
            for row in range(n - 1, -1, -1):
                grid = (triton.cdiv(k, BLOCK_SIZE_K), 1, batch_size)
                linalg_solve_triangular_upper_kernel[grid](
                    A, B, X, n, k, batch_size, row,
                    upper, left, unitriangular,
                    stride_batch_a, stride_batch_b, stride_batch_x,
                    stride_row_a, stride_row_b, stride_row_x,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                )
        else:
            # Lower triangular, solving AX = B
            # Process rows from top to bottom
            for row in range(n):
                grid = (triton.cdiv(k, BLOCK_SIZE_K), 1, batch_size)
                linalg_solve_triangular_lower_kernel[grid](
                    A, B, X, n, k, batch_size, row,
                    upper, left, unitriangular,
                    stride_batch_a, stride_batch_b, stride_batch_x,
                    stride_row_a, stride_row_b, stride_row_x,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                )
    else:
        # Solving XA = B, transpose the problem
        # XA = B -> A^T @ X^T = B^T -> solve for X^T with A^T
        A_T = A.transpose(-2, -1)
        B_T = B.transpose(-2, -1)
        X_T = X.transpose(-2, -1)

        # After transpose, the dimensions swap: n' = k, k' = n
        n_t = k
        k_t = n

        if batch_size > 1:
            stride_batch_a = A_T.stride(0)
            stride_batch_b = B_T.stride(0)
            stride_batch_x = X_T.stride(0)
            stride_row_a = A_T.stride(1)
            stride_row_b = B_T.stride(1)
            stride_row_x = X_T.stride(1)
        else:
            stride_batch_a = 0
            stride_batch_b = 0
            stride_batch_x = 0
            stride_row_a = A_T.stride(0)
            stride_row_b = B_T.stride(0)
            stride_row_x = X_T.stride(0)

        if upper:
            # A is upper triangular, A^T is lower triangular
            for row in range(n_t):
                grid = (triton.cdiv(k_t, BLOCK_SIZE_K), 1, batch_size)
                linalg_solve_triangular_lower_kernel[grid](
                    A_T, B_T, X_T, n_t, k_t, batch_size, row,
                    upper, left, unitriangular,
                    stride_batch_a, stride_batch_b, stride_batch_x,
                    stride_row_a, stride_row_b, stride_row_x,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                )
        else:
            # A is lower triangular, A^T is upper triangular
            for row in range(n_t - 1, -1, -1):
                grid = (triton.cdiv(k_t, BLOCK_SIZE_K), 1, batch_size)
                linalg_solve_triangular_upper_kernel[grid](
                    A_T, B_T, X_T, n_t, k_t, batch_size, row,
                    upper, left, unitriangular,
                    stride_batch_a, stride_batch_b, stride_batch_x,
                    stride_row_a, stride_row_b, stride_row_x,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                )

    return X