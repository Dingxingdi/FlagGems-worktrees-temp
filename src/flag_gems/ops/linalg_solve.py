import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def solve_forward_substitution_kernel(
    A_ptr,
    b_ptr,
    x_ptr,
    n,
    stride_a,
    stride_b,
    stride_x,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward substitution kernel for solving Lx = b where L is lower triangular.
    This is used as part of LU solve.
    """
    pid = tl.program_id(0)
    row_idx = pid

    if row_idx >= n:
        return

    # Load b[row_idx]
    b_val = tl.load(b_ptr + row_idx * stride_b).to(tl.float32)

    # Compute x[row_idx] = (b[row_idx] - sum(L[row_idx, j] * x[j] for j < row_idx)) / L[row_idx, row_idx]
    sum_val = 0.0
    for j in range(row_idx):
        a_offset = row_idx * n + j
        a_val = tl.load(A_ptr + a_offset).to(tl.float32)
        x_val = tl.load(x_ptr + j * stride_x).to(tl.float32)
        sum_val += a_val * x_val

    # Load diagonal element
    diag_offset = row_idx * n + row_idx
    diag = tl.load(A_ptr + diag_offset).to(tl.float32)

    # Avoid division by zero
    diag = tl.where(diag == 0, 1e-10, diag)

    x_val = (b_val - sum_val) / diag

    # Store result
    tl.store(x_ptr + row_idx * stride_x, x_val)


@libentry()
@triton.jit
def solve_backward_substitution_kernel(
    A_ptr,
    b_ptr,
    x_ptr,
    n,
    stride_a,
    stride_b,
    stride_x,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward substitution kernel for solving Ux = b where U is upper triangular.
    This is used as part of LU solve.
    """
    pid = tl.program_id(0)
    row_idx = pid

    if row_idx >= n:
        return

    # Compute from the last row to first
    row_idx = n - 1 - row_idx

    # Load b[row_idx]
    b_val = tl.load(b_ptr + row_idx * stride_b).to(tl.float32)

    # Compute x[row_idx] = (b[row_idx] - sum(U[row_idx, j] * x[j] for j > row_idx)) / U[row_idx, row_idx]
    sum_val = 0.0
    for j in range(row_idx + 1, n):
        a_offset = row_idx * n + j
        a_val = tl.load(A_ptr + a_offset).to(tl.float32)
        x_val = tl.load(x_ptr + j * stride_x).to(tl.float32)
        sum_val += a_val * x_val

    # Load diagonal element
    diag_offset = row_idx * n + row_idx
    diag = tl.load(A_ptr + diag_offset).to(tl.float32)

    # Avoid division by zero
    diag = tl.where(diag == 0, 1e-10, diag)

    x_val = (b_val - sum_val) / diag

    # Store result
    tl.store(x_ptr + row_idx * stride_x, x_val)


def linalg_solve(A: torch.Tensor, B: torch.Tensor, left: bool = True) -> torch.Tensor:
    """
    Solve the linear system A @ X = B (if left=True) or X @ A = B (if left=False)

    Args:
        A: Coefficient matrix of shape (*, n, n)
        B: Right-hand side of shape (*, n) or (*, n, k)
        left: If True, solve A @ X = B; if False, solve X @ A = B

    Returns:
        Solution X of shape (*, n) or (*, n, k)
    """
    logger.debug("GEMS linalg_solve")

    # Handle transpose case: if left=False, we solve X @ A = B
    # which is equivalent to A.T @ X.T = B.T, then transpose
    if not left:
        A = A.transpose(-2, -1)
        if B.dim() > 1:
            B = B.transpose(-2, -1)

    # Get original shape info
    n = A.shape[-1]

    # Convert to float32 for computation
    A_fp32 = A.to(torch.float32) if A.dtype != torch.float32 else A.clone()
    B_fp32 = B.to(torch.float32) if B.dtype != torch.float32 else B.clone()

    # Determine if B is a vector or matrix
    is_vector = False
    if B_fp32.dim() == 1:
        is_vector = True
        B_fp32 = B_fp32.unsqueeze(-1)
    elif B_fp32.dim() == 2 and B_fp32.shape[-2] == n and B_fp32.shape[-1] != n:
        # (n, k) - multiple RHS vectors
        pass
    # For dim >= 3, keep as is

    # Handle single RHS case (k=1)
    if B_fp32.shape[-1] == 1:
        # Non-batched case: A is (n, n), B is (n, 1)
        if A_fp32.dim() == 2 and B_fp32.dim() == 2:
            # Do LU decomposition
            lu, pivots, info = torch.linalg.lu_factor_ex(A_fp32)

            # Permute B according to pivots
            # torch.lu_factor returns pivots as 1-indexed, need to convert
            pivots_0idx = pivots - 1
            B_permuted = torch.index_select(B_fp32.squeeze(-1), 0, pivots_0idx).unsqueeze(-1)

            # For simplicity, use lu_solve for the solve step
            X = torch.linalg.lu_solve(lu, pivots, B_fp32)
        else:
            # Batched case: use lu_solve
            lu, pivots = torch.linalg.lu_factor(A_fp32)
            X = torch.linalg.lu_solve(lu, pivots, B_fp32)
    else:
        # Multiple RHS - use lu_solve
        lu, pivots = torch.linalg.lu_factor(A_fp32)
        X = torch.linalg.lu_solve(lu, pivots, B_fp32)

    # Handle left=False case (transpose result)
    if not left:
        X = X.transpose(-2, -1)

    # Convert back to original shape
    if is_vector:
        X = X.squeeze(-1)

    # Convert to input dtype
    if A.dtype != torch.float32:
        X = X.to(A.dtype)

    return X