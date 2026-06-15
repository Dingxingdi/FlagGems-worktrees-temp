import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def lu_unpack_p_kernel(
    pivots_ptr,
    p_ptr,
    batch_size,
    m,
    k,
    pivots_stride_b,
    pivots_stride_minmn,
    p_stride_b,
    p_stride_m,
    p_stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to construct permutation matrix P from pivots.
    pivots: (batch, k) where k = min(m, n) - 1-indexed pivots
    P: (batch, m, m) - permutation matrix

    Each program computes one element: P[row, col]
    """
    # Compute row and col from program id
    # Grid: (batch_size * m * m,) - exactly covers all elements
    pid = tl.program_id(0)

    batch_id = pid // (m * m)
    remainder = pid % (m * m)
    row = remainder // m
    col = remainder % m

    pivots_base = batch_id * pivots_stride_b
    p_base = batch_id * p_stride_b

    # Find which source row maps to this output row
    # Apply pivots forward: start with row, apply each swap
    current_row = row
    # Iterate over k pivots (k = min(m, n))
    for i in range(k):
        pivot_k = tl.load(pivots_ptr + pivots_base + i * pivots_stride_minmn)
        pivot_row = pivot_k - 1  # 0-indexed
        # If current_row is at position pivot_row, swap to position i
        # If current_row is at position i, swap to position pivot_row
        current_row = tl.where(current_row == pivot_row, i,
                               tl.where(current_row == i, pivot_row, current_row))

    # P[row, col] = 1 if current_row == col, else 0
    p_val = tl.where(current_row == col, 1.0, 0.0)
    tl.store(
        p_ptr + p_base + row * p_stride_m + col * p_stride_n,
        p_val,
    )


@libentry()
@triton.jit
def lu_unpack_l_kernel(
    lu_ptr,
    l_ptr,
    m,
    n,
    k,  # min(m, n)
    lu_stride_b,
    lu_stride_m,
    lu_stride_n,
    l_stride_b,
    l_stride_m,
    l_stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to extract L from packed LU data.
    lu: (batch, m, n) - packed LU data
    L: (batch, m, k) - lower triangular with unit diagonal

    Each program computes one element: L[row, col]
    """
    pid = tl.program_id(0)

    batch_id = pid // (m * k)
    remainder = pid % (m * k)
    row = remainder // k
    col = remainder % k

    lu_base = batch_id * lu_stride_b
    l_base = batch_id * l_stride_b

    # L[row, col] = LU[row, col] if row > col, else (1 if row == col else 0)
    # Load LU value (will be used if row > col)
    lu_val = tl.load(lu_ptr + lu_base + row * lu_stride_m + col * lu_stride_n)

    # Compute the value based on row and col relationship
    val = tl.where(row > col, lu_val,
                   tl.where(row == col, 1.0, 0.0))

    tl.store(l_ptr + l_base + row * l_stride_m + col * l_stride_n, val)


@libentry()
@triton.jit
def lu_unpack_u_kernel(
    lu_ptr,
    u_ptr,
    m,
    n,
    k,  # min(m, n)
    lu_stride_b,
    lu_stride_m,
    lu_stride_n,
    u_stride_b,
    u_stride_m,
    u_stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to extract U from packed LU data.
    lu: (batch, m, n) - packed LU data
    U: (batch, k, n) - upper triangular with diagonal

    Each program computes one element: U[row, col]
    """
    pid = tl.program_id(0)

    batch_id = pid // (k * n)
    remainder = pid % (k * n)
    row = remainder // n
    col = remainder % n

    lu_base = batch_id * lu_stride_b
    u_base = batch_id * u_stride_b

    # U[row, col] = LU[row, col] if row <= col, else 0
    lu_val = tl.load(lu_ptr + lu_base + row * lu_stride_m + col * lu_stride_n)

    val = tl.where(row <= col, lu_val, 0.0)

    tl.store(u_ptr + u_base + row * u_stride_m + col * u_stride_n, val)


def lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    """
    Unpacks the LU decomposition into P, L, U matrices.

    Args:
        LU_data: Tensor of shape (..., m, n) containing the packed LU factorization
        LU_pivots: Tensor of shape (..., min(m, n)) containing the pivots (1-indexed)
        unpack_data: If True, unpack L and U. If False, return empty tensors.
        unpack_pivots: If True, unpack P. If False, return empty tensor.

    Returns:
        Tuple of (P, L, U) where:
            P: (..., m, m) permutation matrix
            L: (..., m, min(m, n)) lower triangular with unit diagonal
            U: (..., min(m, n), n) upper triangular
    """
    # Get shapes
    lu_shape = LU_data.shape
    pivots_shape = LU_pivots.shape

    # Handle batch dimensions
    batch_dims = lu_shape[:-2]
    m, n = lu_shape[-2], lu_shape[-1]
    k = min(m, n)  # min(m, n)

    logger.debug(
        "GEMS LU_UNPACK, shape: %s, m: %s, n: %s, k: %s",
        lu_shape,
        m,
        n,
        k,
    )

    # Prepare outputs
    device = LU_data.device
    dtype = LU_data.dtype

    # Handle batch size
    if len(batch_dims) > 0:
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim
    else:
        batch_size = 1

    if unpack_pivots:
        # P has shape (..., m, m)
        P = torch.zeros(*batch_dims, m, m, device=device, dtype=dtype)

        BLOCK_SIZE = 128  # Dummy, not used
        grid = (batch_size * m * m,)

        lu_unpack_p_kernel[grid](
            LU_pivots,
            P,
            batch_size,
            m,
            k,
            LU_pivots.stride(-2) if len(pivots_shape) > 1 else 0,
            LU_pivots.stride(-1) if len(pivots_shape) > 0 else 0,
            P.stride(-3) if len(batch_dims) > 0 else 0,
            P.stride(-2),
            P.stride(-1),
            BLOCK_SIZE,
        )
    else:
        # Return empty tensor
        P = torch.empty(0, device=device, dtype=dtype)

    if unpack_data:
        # L has shape (..., m, k)
        L = torch.empty(*batch_dims, m, k, device=device, dtype=dtype)
        # U has shape (..., k, n)
        U = torch.empty(*batch_dims, k, n, device=device, dtype=dtype)

        BLOCK_SIZE = 128

        # Extract L (lower triangular)
        grid_l = (batch_size * m * k,)

        lu_unpack_l_kernel[grid_l](
            LU_data,
            L,
            m,
            n,
            k,
            LU_data.stride(-3) if len(lu_shape) > 2 else 0,
            LU_data.stride(-2),
            LU_data.stride(-1),
            L.stride(-3) if len(batch_dims) > 0 else 0,
            L.stride(-2),
            L.stride(-1),
            BLOCK_SIZE,
        )

        # Extract U (upper triangular)
        grid_u = (batch_size * k * n,)

        lu_unpack_u_kernel[grid_u](
            LU_data,
            U,
            m,
            n,
            k,
            LU_data.stride(-3) if len(lu_shape) > 2 else 0,
            LU_data.stride(-2),
            LU_data.stride(-1),
            U.stride(-3) if len(batch_dims) > 0 else 0,
            U.stride(-2),
            U.stride(-1),
            BLOCK_SIZE,
        )
    else:
        # Return empty tensors
        L = torch.empty(0, device=device, dtype=dtype)
        U = torch.empty(0, device=device, dtype=dtype)

    return (P, L, U)