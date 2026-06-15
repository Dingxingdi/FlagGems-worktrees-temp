import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def cholesky_inverse_mm_kernel(
    Inv,
    Out,
    n,
    stride_bn,
    stride_n,
    stride_n2,
    upper: tl.constexpr,
):
    """
    Compute (Inv^T @ Inv) for lower triangular or (Inv @ Inv^T) for upper triangular.
    This computes A^{-1} = (L^{-1})^T @ L^{-1} for lower, or L^{-1} @ (L^{-1})^T for upper.
    Grid: (batch_size, n, n)
    """
    # Get batch index, row and column indices
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_idx = tl.program_id(2)

    if row_idx >= n or col_idx >= n:
        return

    # Calculate the base offset for this batch element
    base_offset = batch_idx * stride_bn

    acc = 0.0
    for k in range(n):
        if upper:
            # C = A @ A^T: C[row, col] = sum_k A[row, k] * A[col, k]
            a_offs_r = base_offset + row_idx * stride_n + k * stride_n2
            a_offs_c = base_offset + col_idx * stride_n + k * stride_n2
        else:
            # C = A^T @ A: C[row, col] = sum_k A[k, row] * A[k, col]
            a_offs_r = base_offset + k * stride_n + row_idx * stride_n2
            a_offs_c = base_offset + k * stride_n + col_idx * stride_n2

        a_val_r = tl.load(Inv + a_offs_r)
        a_val_c = tl.load(Inv + a_offs_c)
        acc += a_val_r * a_val_c

    off = base_offset + row_idx * stride_n + col_idx * stride_n2
    tl.store(Out + off, acc)


def triangular_inverse_cpu(L, upper=False):
    """
    Compute the inverse of a triangular matrix on CPU.
    Input L has shape (*, n, n) where * is batch dimensions.

    For lower triangular L: solve L @ X = I, X = L^{-1}
    For upper triangular U: solve U^T @ X = I (equivalent to computing (U^-1)^T)
    """
    n = L.shape[-1]
    batch_shape = L.shape[:-2]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    L_inv = torch.zeros_like(L)

    if upper:
        # For upper triangular U:
        # Compute inverse of L = U^T (lower triangular) and transpose
        L_transpose = L.transpose(-2, -1)

        L_transpose_flat = L_transpose.reshape(batch_size, n, n)
        L_inv_flat = L_inv.reshape(batch_size, n, n)
        for b in range(batch_size):
            L_b = L_transpose_flat[b]
            L_inv_b = L_inv_flat[b]
            for j in range(n):
                for i in range(j, n):
                    if i == j:
                        L_inv_b[i, j] = 1.0 / L_b[i, j].item()
                    else:
                        s = 0.0
                        for k in range(j, i):
                            s += L_b[i, k].item() * L_inv_b[k, j].item()
                        L_inv_b[i, j] = -s / L_b[i, i].item()
        L_inv = L_inv_flat.reshape(L.shape)

        # Transpose to get U^{-1}
        L_inv = L_inv.transpose(-2, -1)
    else:
        # For lower triangular L: solve L @ X = I
        L_flat = L.reshape(batch_size, n, n)
        L_inv_flat = L_inv.reshape(batch_size, n, n)
        for b in range(batch_size):
            L_b = L_flat[b]
            L_inv_b = L_inv_flat[b]
            for j in range(n):
                for i in range(j, n):
                    if i == j:
                        L_inv_b[i, j] = 1.0 / L_b[i, j].item()
                    else:
                        s = 0.0
                        for k in range(j, i):
                            s += L_b[i, k].item() * L_inv_b[k, j].item()
                        L_inv_b[i, j] = -s / L_b[i, i].item()
        L_inv = L_inv_flat.reshape(L.shape)

    return L_inv


def cholesky_inverse(L, upper=False):
    """
    Computes the inverse of a complex Hermitian or real symmetric
    positive-definite matrix given its Cholesky decomposition.

    Args:
        L: Tensor of shape (*, n, n) where * is zero or more batch dimensions
           consisting of lower or upper triangular Cholesky decompositions.
        upper: If True, L is upper triangular; if False, L is lower triangular.

    Returns:
        The inverse of the original matrix.
    """
    logger.debug("GEMS CHOLESKY_INVERSE")

    # Handle the case of no batch dimensions (single matrix)
    if L.dim() == 2:
        L = L.unsqueeze(0)
        single_matrix = True
    else:
        single_matrix = False

    # Get matrix dimensions
    *batch_dims, n, m = L.shape
    if n != m:
        raise ValueError("Input matrix must be square")

    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    # Create output tensor
    output_shape = list(L.shape)
    output = torch.empty(output_shape, dtype=L.dtype, device=L.device)

    # Make input contiguous for efficient access
    L = L.contiguous()

    with torch_device_fn.device(L.device):
        # Step 1: Compute triangular inverse (L_inv) using CPU kernel
        # Move to CPU for sequential computation
        L_cpu = L.cpu()
        L_inv_cpu = triangular_inverse_cpu(L_cpu, upper=upper)
        L_inv = L_inv_cpu.to(L.device)

        # Step 2: Compute L_inv^T @ L_inv (or L_inv @ L_inv^T for upper) using Triton
        # For lower: A^{-1} = (L^{-1})^T @ L^{-1}
        # For upper: A^{-1} = (U^{-1}) @ (U^{-1})^T
        grid = (batch_size, n, n)
        cholesky_inverse_mm_kernel[grid](
            L_inv,
            output,
            n,
            L_inv.stride(0),
            L_inv.stride(1),
            L_inv.stride(2),
            upper,
        )

    if single_matrix:
        output = output.squeeze(0)

    return output