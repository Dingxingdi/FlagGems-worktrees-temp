import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def sparse_sampled_addmm_kernel(
    result_values_ptr,
    mat1_ptr,
    mat2_ptr,
    input_crow_indices_ptr,
    input_col_indices_ptr,
    input_values_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    nnz,
    stride_mat1_m,
    stride_mat1_k,
    stride_mat2_k,
    stride_mat2_n,
    stride_result_n,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the position of this thread in the sparse tensor
    pid = tle.program_id(0)
    num_positions = nnz

    if pid >= num_positions:
        return

    # Load sparse tensor info
    # Each thread processes one non-zero element
    row_start = tl.load(input_crow_indices_ptr + pid)
    # For the last row, we need to handle differently
    if pid + 1 < nnz:
        row_end = tl.load(input_crow_indices_ptr + pid + 1)
    else:
        row_end = M
    row = row_start

    col = tl.load(input_col_indices_ptr + pid)
    input_val = tl.load(input_values_ptr + pid)

    # Compute the dot product for this (row, col) position
    # result[row, col] = mat1[row, :] @ mat2[:, col]
    accumulator = 0.0
    for k in range(0, K):
        mat1_val = tl.load(mat1_ptr + row * stride_mat1_m + k * stride_mat1_k)
        mat2_val = tl.load(mat2_ptr + k * stride_mat2_k + col * stride_mat2_n)
        accumulator += mat1_val * mat2_val

    # Apply the formula: out = alpha * dense_result + beta * input
    result_val = alpha * accumulator + beta * input_val

    # Store the result
    tl.store(result_values_ptr + pid, result_val)


def sparse_sampled_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    """Performs a matrix multiplication of dense matrices mat1 and mat2 at the
    locations specified by the sparsity pattern of input. The matrix input is
    added to the final result.

    Mathematically this performs the following operation:
        out = alpha * (mat1 @ mat2) * spy(input) + beta * input

    Args:
        input: a sparse CSR matrix of shape (m, n)
        mat1: a dense matrix of shape (m, k)
        mat2: a dense matrix of shape (k, n)
        beta: multiplier for input
        alpha: multiplier for mat1 @ mat2

    Returns:
        A sparse CSR tensor with the same sparsity pattern as input
    """
    assert input.layout == torch.sparse_csr, "input must be a sparse CSR tensor"
    assert mat1.shape[0] == input.shape[0], "mat1 must have same number of rows as input"
    assert mat2.shape[1] == input.shape[1], "mat2 must have same number of columns as input"
    assert mat1.shape[1] == mat2.shape[0], "mat1 and mat2 must be compatible for matmul"

    M, N = input.shape
    K = mat1.shape[1]
    nnz = input._nnz()

    logger.debug(
        "GEMS SPARSE_SAMPLED_ADDMM, [shape info]: M=%s, N=%s, K=%s, nnz=%s, "
        "alpha=%s, beta=%s",
        M, N, K, nnz, alpha, beta
    )

    # Get the CSR format info from input
    crow_indices = input.crow_indices()
    col_indices = input.col_indices()
    input_values = input.values()

    # Allocate output sparse tensor
    result_values = torch.empty(nnz, dtype=input.dtype, device=input.device)

    # Define grid - one thread per non-zero element
    grid = (nnz,)

    sparse_sampled_addmm_kernel[grid](
        result_values,
        mat1,
        mat2,
        crow_indices,
        col_indices,
        input_values,
        alpha,
        beta,
        M,
        N,
        K,
        nnz,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        result_values.stride(0),
        BLOCK_SIZE=128,
    )

    # Construct the output sparse CSR tensor
    output = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        result_values,
        size=input.shape,
        dtype=input.dtype,
        device=input.device
    )

    return output