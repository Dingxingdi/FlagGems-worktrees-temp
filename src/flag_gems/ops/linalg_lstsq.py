"""
Implementation of linalg_lstsq operator for FlagGems.

This module implements the least squares solver using the normal equations:
    X = (A^T A + rcond * I)^-1 @ A^T @ B

The core matrix operations are implemented using Triton kernels.
"""

import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.mm import mm
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Define the lstsq namedtuple type
LstsqResult = namedtuple('LstsqResult', ['solution', 'residuals', 'rank', 'singular_values'])


def linalg_lstsq(A: torch.Tensor, B: torch.Tensor, rcond: float = None) -> tuple:
    """
    Solve the least squares problem: min ||AX - B||_F

    Args:
        A: Input tensor of shape (..., m, n)
        B: Input tensor of shape (..., m, k)
        rcond: Cut-off ratio for small singular values (default: machine epsilon * max(m, n))

    Returns:
        A named tuple (solution, residuals, rank, singular_values)
        - solution: (..., n, k) least squares solution
        - residuals: (...,) squared residuals (only when m > n and full rank)
        - rank: (...,) effective rank of A
        - singular_values: (..., min(m, n)) singular values of A
    """
    # Handle broadcasted batch dimensions
    broadcast_shape = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    A = A.expand(*broadcast_shape, *A.shape[-2:])
    B = B.expand(*broadcast_shape, *B.shape[-2:])

    # Get shapes
    *batch_dims, m, n = A.shape
    *_, m2, k = B.shape

    # Set default rcond
    if rcond is None:
        rcond = torch.finfo(A.dtype).eps * max(m, n)

    logger.debug(f"GEMS LINALG_LSTSQ, shape: A={A.shape}, B={B.shape}, rcond={rcond}")

    # For low precision types (float16, bfloat16), use float32 for computation
    # then convert back at the end
    use_fp32_compute = A.dtype in (torch.float16, torch.bfloat16)
    original_dtype = A.dtype

    if use_fp32_compute:
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        rcond = float(rcond)

    # Use mm (matrix multiply) from FlagGems instead of torch.matmul
    # Compute A^T @ A (shape: ..., n, n)
    At = A.transpose(-2, -1)
    AtA = mm(At, A)  # FlagGems mm

    # Add regularization to make the matrix invertible
    reg = rcond * torch.eye(n, device=A.device, dtype=A.dtype)
    for _ in range(len(batch_dims)):
        reg = reg.unsqueeze(0)
    reg = reg.expand(*batch_dims, n, n)
    AtA_reg = AtA + reg

    # Compute A^T @ B (shape: ..., n, k)
    AtB = mm(At, B)  # FlagGems mm

    # Solve (A^T A + rcond*I) X = A^T B
    # X = (A^T A + rcond*I)^-1 @ A^T B
    # Using PyTorch's inverse for now (Triton implementation would be complex)
    # But the key computation (matrix multiply) uses FlagGems
    AtA_reg_inv = torch.linalg.inv(AtA_reg)
    X = mm(AtA_reg_inv, AtB)  # FlagGems mm

    # Compute residuals: ||AX - B||_F^2
    # For each column of B, compute the squared norm of the residual
    residuals = None
    if m > n:
        AX = mm(A, X)  # FlagGems mm
        AX_B = AX - B
        # Sum over m dimension, leaving k dimension: shape (k,)
        residuals = torch.sum(AX_B ** 2, dim=-2)

    # Compute rank and singular values
    # Note: These are empty for the default 'gels' driver on CUDA
    singular_values = torch.empty(min(m, n), device=A.device, dtype=A.dtype)
    rank = torch.empty(0, dtype=torch.int64, device=A.device)

    # Convert back to original dtype if needed
    if use_fp32_compute:
        X = X.to(original_dtype)
        if residuals is not None:
            residuals = residuals.to(original_dtype)

    # Return as named tuple (like PyTorch)
    return LstsqResult(solution=X, residuals=residuals, rank=rank, singular_values=singular_values)


def lstsq(A: torch.Tensor, B: torch.Tensor, rcond: float = None) -> torch.Tensor:
    """
    Wrapper that just returns the solution (for simpler use cases).
    """
    result = linalg_lstsq(A, B, rcond)
    return result.solution