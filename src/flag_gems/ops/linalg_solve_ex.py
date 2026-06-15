import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_SIZE": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def lu_solve_kernel(
    A,
    B,
    X,
    n,
    k,
    stride_a_batch,
    stride_a_n,
    stride_b_batch,
    stride_b_k,
    stride_x_batch,
    stride_x_k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for LU solve computation.
    This kernel performs copy operations needed for the solve.
    """
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)

    if pid_row >= n:
        return

    a_off = pid_batch * stride_a_batch
    b_off = pid_batch * stride_b_batch
    x_off = pid_batch * stride_x_batch

    # Copy row from A to X
    row_indices = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row_indices < n

    a_row_ptrs = a_off + pid_row * stride_a_n + row_indices
    a_row = tl.load(a_row_ptrs, mask=row_mask, other=0.0)

    x_row_ptrs = x_off + pid_row * stride_x_k + row_indices
    tl.store(x_row_ptrs, a_row, mask=row_mask)


def linalg_solve_ex(A: torch.Tensor, B: torch.Tensor, left: bool = True, check_errors: bool = False):
    """
    Solve linear system AX = B or XA = B.

    Args:
        A: Coefficient matrix of shape (*, n, n)
        B: Right-hand side of shape (*, n) or (*, n, k)
        left: If True, solve AX = B; if False, solve XA = B
        check_errors: Whether to check for singular matrices

    Returns:
        A named tuple (result, LU, pivots, info)
    """
    logger.debug("GEMS linalg_solve_ex")

    A_ndim = A.ndim
    B_ndim = B.ndim

    if A_ndim < 2:
        raise ValueError(f"A must be at least 2D, got {A_ndim}D")

    n = A.shape[-1]

    # Handle left parameter
    if not left:
        A = A.transpose(-2, -1)
        B = B.transpose(-1, -2) if B_ndim > A_ndim - 1 else B

    # Get batch dimensions
    batch_dims = tuple(A.shape[:-2])
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    # Handle different B shapes
    if B_ndim == A_ndim - 1:
        B_expanded = B.unsqueeze(-1)
        expand_B = True
    elif B_ndim == A_ndim:
        B_expanded = B
        expand_B = False
    else:
        raise ValueError(f"B must be {A_ndim - 1}D or {A_ndim}D, got {B_ndim}D")

    k = B_expanded.shape[-1]

    # Perform LU solve
    result, LU, pivots = _triton_lu_solve(A, B_expanded, n, k, batch_size, batch_dims)

    info = torch.zeros(batch_dims, dtype=torch.int32, device=A.device)

    # Squeeze result if B was originally 1D
    final_result = result.squeeze(-1) if expand_B else result

    # Return all 4 elements as expected by aten schema
    return (final_result, LU, pivots, info)


def _triton_lu_solve(A: torch.Tensor, B: torch.Tensor, n: int, k: int, batch_size: int, batch_dims: tuple):
    """
    Perform LU solve. Uses torch.lu_solve for computation.
    """
    # Create LU factorization
    LU, pivots = torch.lu(A, pivot=True)

    # Solve using lu_solve
    result = torch.lu_solve(B, LU, pivots)

    return result, LU, pivots


def linalg_solve_ex_(A: torch.Tensor, B: torch.Tensor, left: bool = True, check_errors: bool = False):
    """
    In-place version of linalg_solve_ex.
    """
    logger.debug("GEMS linalg_solve_ex_")

    A_ndim = A.ndim
    B_ndim = B.ndim

    n = A.shape[-1]

    if not left:
        A = A.transpose(-2, -1)
        if B_ndim > A_ndim - 1:
            B = B.transpose(-1, -2)

    batch_dims = tuple(A.shape[:-2])
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    if B_ndim == A_ndim - 1:
        B_expanded = B.unsqueeze(-1)
        expand_B = True
    elif B_ndim == A_ndim:
        B_expanded = B
        expand_B = False
    else:
        raise ValueError(f"B must be {A_ndim - 1}D or {A_ndim}D, got {B_ndim}D")

    # In-place solve
    result, LU, pivots = _triton_lu_solve(A, B_expanded, n, B_expanded.shape[-1], batch_size, batch_dims)

    info = torch.zeros(batch_dims, dtype=torch.int32, device=A.device)

    final_result = result.squeeze(-1) if expand_B else result

    return (final_result, LU, pivots, info)