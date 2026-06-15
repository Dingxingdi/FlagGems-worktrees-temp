import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def qr_normalize_column_kernel(
    A_ptr,
    Q_ptr,
    col,
    M,
    K,
    stride_am,
    stride_an,
    stride_qm,
    stride_qk,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to normalize a column to get Q column."""
    pid = tl.program_id(0)

    if pid > 0:
        return

    offs_m = tl.arange(0, BLOCK_SIZE)
    mask_m = offs_m < M

    # Load the column from A (already orthogonalized)
    a_col = tl.load(A_ptr + offs_m * stride_am + col * stride_an, mask=mask_m, other=0.0)

    # Compute 2-norm
    norm_sq = tl.sum(a_col * a_col)
    norm = tl.sqrt(norm_sq + 1e-10)

    # Normalize to get Q column
    q_col = a_col / norm

    # Store Q column
    tl.store(Q_ptr + offs_m * stride_qm + col * stride_qk, q_col, mask=mask_m)


@libentry()
@triton.jit
def qr_orthogonalize_against_prev_kernel(
    A_ptr,
    Q_ptr,
    col,
    M,
    K,
    stride_am,
    stride_an,
    stride_qm,
    stride_qk,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to orthogonalize column col against all previous Q columns."""
    pid = tl.program_id(0)

    if pid > 0:
        return

    offs_m = tl.arange(0, BLOCK_SIZE)
    mask_m = offs_m < M

    # Load column col from A
    a_col = tl.load(A_ptr + offs_m * stride_am + col * stride_an, mask=mask_m, other=0.0)

    # Orthogonalize against all previous Q columns (j < col)
    for j in range(col):
        # Load Q column j
        q_j = tl.load(Q_ptr + offs_m * stride_qm + j * stride_qk, mask=mask_m, other=0.0)

        # Compute dot product
        r_jc = tl.sum(q_j * a_col)

        # Orthogonalize: a_col = a_col - r_jc * Q[:, j]
        a_col = a_col - r_jc * q_j

    # Store the orthogonalized column back
    tl.store(A_ptr + offs_m * stride_am + col * stride_an, a_col, mask=mask_m)


@libentry()
@triton.jit
def qr_orthogonalize_remaining_kernel(
    A_ptr,
    Q_ptr,
    col,
    M,
    K,
    N,
    stride_am,
    stride_an,
    stride_qm,
    stride_qk,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to orthogonalize columns j > col against Q[:, col]."""
    pid = tl.program_id(0)

    if pid > 0:
        return

    offs_m = tl.arange(0, BLOCK_SIZE)
    mask_m = offs_m < M

    # Load Q column col
    q_col = tl.load(Q_ptr + offs_m * stride_qm + col * stride_qk, mask=mask_m, other=0.0)

    # Orthogonalize each column j > col against Q[:, col]
    for j in range(col + 1, K):
        # Load A column j
        a_j = tl.load(A_ptr + offs_m * stride_am + j * stride_an, mask=mask_m, other=0.0)

        # Compute dot product
        r_cj = tl.sum(q_col * a_j)

        # Orthogonalize: a_j = a_j - r_cj * Q[:, col]
        a_j_new = a_j - r_cj * q_col

        # Store back
        tl.store(A_ptr + offs_m * stride_am + j * stride_an, a_j_new, mask=mask_m)


@libentry()
@triton.jit
def qr_compute_r_offdiag_kernel(
    A_orig_ptr,
    Q_ptr,
    R_ptr,
    col,
    M,
    N,
    K,
    stride_aorig_m,
    stride_aorig_n,
    stride_qm,
    stride_qk,
    stride_rk,
    stride_rn,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to compute off-diagonal R elements using original A."""
    pid = tl.program_id(0)

    if pid > 0:
        return

    offs_m = tl.arange(0, BLOCK_SIZE)
    mask_m = offs_m < M

    # Load Q column col
    q_col = tl.load(Q_ptr + offs_m * stride_qm + col * stride_qk, mask=mask_m, other=0.0)

    # Compute R[col, j] for j > col
    for j in range(col + 1, N):
        # Load original A column j
        a_j = tl.load(A_orig_ptr + offs_m * stride_aorig_m + j * stride_aorig_n, mask=mask_m, other=0.0)

        # Compute dot product
        r_cj = tl.sum(q_col * a_j)

        # Store R[col, j]
        tl.store(R_ptr + col * stride_rn + j, r_cj)


def linalg_qr(A, mode="reduced"):
    """
    Computes the QR decomposition of a matrix using Triton kernels.

    Args:
        A: Input tensor of shape (..., M, N)
        mode: One of 'reduced', 'complete', 'r'

    Returns:
        (Q, R) where Q is orthogonal and R is upper triangular
    """
    logger.debug("GEMS LINALG_QR")

    if A.dim() < 2:
        raise ValueError("linalg_qr: expected at least 2D tensor")

    original_shape = A.shape
    if A.dim() > 2:
        batch_shape = original_shape[:-2]
        M = original_shape[-2]
        N = original_shape[-1]
        A = A.reshape(-1, M, N)
        is_batched = True
    else:
        batch_shape = ()
        M = A.shape[-2]
        N = A.shape[-1]
        A = A.unsqueeze(0)  # Add batch dimension for consistent processing
        is_batched = False

    K = min(M, N)

    if mode == "reduced":
        Q_shape = (M, K)
        R_shape = (K, N)
    elif mode == "complete":
        Q_shape = (M, M)
        R_shape = (M, N)
    elif mode == "r":
        Q_shape = (0,)
        R_shape = (K, N)
    else:
        raise ValueError(f"linalg_qr: invalid mode '{mode}'")

    if mode == "r":
        Q = torch.empty(Q_shape, dtype=A.dtype, device=A.device)
        R = torch.empty(R_shape, dtype=A.dtype, device=A.device)
        return (Q, R)

    num_matrices = A.shape[0]
    Q_all = torch.empty((num_matrices,) + Q_shape, dtype=A.dtype, device=A.device)
    R_all = torch.empty((num_matrices,) + R_shape, dtype=A.dtype, device=A.device)

    BLOCK_SIZE = 128

    for b in range(num_matrices):
        A_mat = A[b]
        # Keep a copy of original A for computing R
        A_orig = A_mat.contiguous().clone()
        A_work = A_mat.contiguous().clone()
        Q_mat = torch.empty(Q_shape, dtype=A.dtype, device=A.device)
        R_mat = torch.zeros(R_shape, dtype=A.dtype, device=A.device)

        # Modified Gram-Schmidt using Triton kernels
        for i in range(K):
            # Step 1: Orthogonalize column i against previous Q columns
            grid = (1,)
            qr_orthogonalize_against_prev_kernel[grid](
                A_work,
                Q_mat,
                i,
                M,
                K,
                A_work.stride(0),
                A_work.stride(1),
                Q_mat.stride(0),
                Q_mat.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )

            # Step 2: Compute R[i,i] from the orthogonalized column
            a_col_ortho = A_work[:, i]
            r_diag = torch.sqrt(torch.sum(a_col_ortho * a_col_ortho) + 1e-10)
            R_mat[i, i] = r_diag

            # Step 3: Normalize to get Q[:, i]
            grid = (1,)
            qr_normalize_column_kernel[grid](
                A_work,
                Q_mat,
                i,
                M,
                K,
                A_work.stride(0),
                A_work.stride(1),
                Q_mat.stride(0),
                Q_mat.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )

            # Step 4: Orthogonalize remaining columns against Q[:, i]
            grid = (1,)
            qr_orthogonalize_remaining_kernel[grid](
                A_work,
                Q_mat,
                i,
                M,
                K,
                N,
                A_work.stride(0),
                A_work.stride(1),
                Q_mat.stride(0),
                Q_mat.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )

        # Compute R off-diagonal elements using Python
        # R[i,j] = Q[:,i] · A_original[:,j] for j > i
        for i in range(K):
            for j in range(i + 1, N):
                r_ij = torch.sum(Q_mat[:, i] * A_mat[:, j])
                R_mat[i, j] = r_ij

        Q_all[b] = Q_mat
        R_all[b] = R_mat

    if is_batched:
        Q_all = Q_all.reshape(*batch_shape, *Q_shape)
        R_all = R_all.reshape(*batch_shape, *R_shape)
    else:
        # Single matrix case - squeeze the batch dimension
        Q_all = Q_all.squeeze(0)
        R_all = R_all.squeeze(0)

    return (Q_all, R_all)


def linalg_qr_(A, mode="reduced"):
    """In-place QR not supported."""
    raise RuntimeError("linalg_qr_: in-place QR decomposition is not supported")