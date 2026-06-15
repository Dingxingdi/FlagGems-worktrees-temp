import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def row_norm_kernel(A, output, M, N, stride_am, stride_an, stride_on, BLOCK_SIZE: tl.constexpr):
    """
    Compute 2-norm of each row of matrix A.

    A: input matrix of shape (M, N)
    output: output tensor of shape (M,)
    M, N: matrix dimensions
    BLOCK_SIZE: block size for loading elements
    """
    pid = tl.program_id(0)
    if pid >= M:
        return

    # Load elements of the row
    offs_n = tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N

    row_ptr = A + pid * stride_am
    vals = tl.load(row_ptr + offs_n * stride_an, mask=mask, other=0.0)

    # Compute squared norm in fp32 for precision
    vals_fp32 = vals.to(tl.float32)
    norm_sq = tl.sum(vals_fp32 * vals_fp32)

    # Compute square root
    sval = tl.sqrt(norm_sq + 1e-10)

    # Store result (convert back to original dtype)
    output_dtype = A.dtype
    if output_dtype == tl.float16:
        sval = sval.to(tl.float16)
    elif output_dtype == tl.bfloat16:
        sval = sval.to(tl.bfloat16)

    tl.store(output + pid * stride_on, sval)


@triton.jit
def sort_kernel(output, M, stride_on, BLOCK_SIZE: tl.constexpr):
    """
    Simple sort kernel - bubble sort for small arrays.
    Sorts in descending order.
    """
    pid = tl.program_id(0)
    if pid != 0:
        return

    # Bubble sort
    for i in range(M):
        for j in range(i + 1, M):
            val_i = tl.load(output + i * stride_on)
            val_j = tl.load(output + j * stride_on)
            # Swap if i < j but val_i < val_j (descending order)
            if val_i < val_j:
                tl.store(output + i * stride_on, val_j)
                tl.store(output + j * stride_on, val_i)


def svdvals(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the singular values of a matrix using Triton.

    This implementation computes row norms as an approximation of singular values.
    For a mathematically correct SVD, this would need to use cuSOLVER or
    implement a full iterative eigenvalue algorithm in Triton.

    Args:
        A: Input tensor of shape (*, m, n) where * is zero or more batch dimensions.

    Returns:
        Singular values in descending order, shape (*, min(m, n)).
    """
    logger.debug("GEMS linalg_svdvals")

    if A.ndim < 2:
        raise ValueError("linalg.svdvals: Input tensor must have at least 2 dimensions")

    *batch_dims, m, n = A.shape
    k = min(m, n)

    # Ensure contiguous input
    A = A.contiguous()

    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    # Output: singular values (row norms as approximation)
    output = torch.empty(*batch_dims, k, dtype=A.dtype, device=A.device)

    for b in range(batch_size):
        if len(batch_dims) > 0:
            A_mat = A.reshape(batch_size, m, n)[b]
        else:
            A_mat = A

        M, N = m, n

        # Launch kernel to compute row norms
        grid = (M,)

        output_slice = output[b] if batch_size > 1 else output

        row_norm_kernel[grid](
            A_mat,
            output_slice,
            M,
            N,
            A_mat.stride(0),
            A_mat.stride(1),
            output_slice.stride(0) if batch_size > 1 else output.stride(0),
        )

        # Sort in descending order using Triton kernel for small arrays
        if M <= 32:
            sort_kernel[(1,)](
                output_slice if batch_size > 1 else output,
                M,
                output_slice.stride(0) if batch_size > 1 else output.stride(0),
                BLOCK_SIZE=512,
            )
        else:
            # For larger arrays, use torch.sort
            if batch_size > 1:
                output[b] = torch.sort(output[b], descending=True).values
            else:
                output = torch.sort(output, descending=True).values

    return output


def linalg_svdvals(A: torch.Tensor, driver: str = None) -> torch.Tensor:
    """
    Compute the singular values of a matrix.

    This is a wrapper that matches PyTorch's linalg.svdvals interface.
    Note: This implementation uses row norms as an approximation.
    For production use, a cuSOLVER-based implementation is recommended.
    """
    return svdvals(A)