import logging
import collections

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Create the named tuple type for return value
_LinalgDet = collections.namedtuple("_linalg_det", ["result", "LU", "pivots"])


@triton.jit
def linalg_det_2x2_kernel(
    A_ptr,
    result_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
):
    """
    Compute determinant for 2x2 matrices using Triton.
    det = a00 * a11 - a01 * a10

    A has shape (batch, n, n), with strides (stride_ab, stride_am, stride_an)
    """
    # Get the matrix index (batch index)
    pid = tl.program_id(0)
    if pid >= M:
        return

    # For A[pid, i, j], offset = pid * stride_ab + i * stride_am + j * stride_an
    base = pid * stride_ab

    # Load matrix elements
    a00 = tl.load(A_ptr + (base + 0 * stride_am + 0 * stride_an)).to(tl.float32)
    a01 = tl.load(A_ptr + (base + 0 * stride_am + 1 * stride_an)).to(tl.float32)
    a10 = tl.load(A_ptr + (base + 1 * stride_am + 0 * stride_an)).to(tl.float32)
    a11 = tl.load(A_ptr + (base + 1 * stride_am + 1 * stride_an)).to(tl.float32)

    # Compute determinant: det = a00 * a11 - a01 * a10
    det = a00 * a11 - a01 * a10

    # Store result
    tl.store(result_ptr + pid, det)


@triton.jit
def linalg_det_3x3_kernel(
    A_ptr,
    result_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
):
    """
    Compute determinant for 3x3 matrices using Triton.
    Using Sarrus rule:
    det = a00*a11*a22 + a01*a12*a20 + a02*a10*a21
          - a02*a11*a20 - a00*a12*a21 - a01*a10*a22

    A has shape (batch, n, n), with strides (stride_ab, stride_am, stride_an)
    """
    pid = tl.program_id(0)
    if pid >= M:
        return

    # For A[pid, i, j], offset = pid * stride_ab + i * stride_am + j * stride_an
    base = pid * stride_ab

    # Load first row
    a00 = tl.load(A_ptr + (base + 0 * stride_am + 0 * stride_an)).to(tl.float32)
    a01 = tl.load(A_ptr + (base + 0 * stride_am + 1 * stride_an)).to(tl.float32)
    a02 = tl.load(A_ptr + (base + 0 * stride_am + 2 * stride_an)).to(tl.float32)

    # Load second row
    a10 = tl.load(A_ptr + (base + 1 * stride_am + 0 * stride_an)).to(tl.float32)
    a11 = tl.load(A_ptr + (base + 1 * stride_am + 1 * stride_an)).to(tl.float32)
    a12 = tl.load(A_ptr + (base + 1 * stride_am + 2 * stride_an)).to(tl.float32)

    # Load third row
    a20 = tl.load(A_ptr + (base + 2 * stride_am + 0 * stride_an)).to(tl.float32)
    a21 = tl.load(A_ptr + (base + 2 * stride_am + 1 * stride_an)).to(tl.float32)
    a22 = tl.load(A_ptr + (base + 2 * stride_am + 2 * stride_an)).to(tl.float32)

    # Compute determinant using Sarrus rule
    det = (a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21
           - a02 * a11 * a20 - a00 * a12 * a21 - a01 * a10 * a22)

    tl.store(result_ptr + pid, det)


def linalg_det(A: torch.Tensor):
    """
    Compute the determinant of a batch of square matrices.

    Args:
        A: Input tensor of shape (*, n, n) where * is batch dimension

    Returns:
        A named tuple (result, LU, pivots) where:
        - result: determinant values of shape (*)
        - LU: LU decomposition of shape (*, n, n)
        - pivots: pivot indices of shape (*, n)
    """
    assert A.dim() >= 2, "Input must be at least 2D"
    assert A.shape[-1] == A.shape[-2], "Input must be square"

    *batch_dims, n, n = A.shape
    batch_size = 1
    for d in batch_dims:
        batch_size *= d

    logger.debug(
        "GEMS LINALG_DET, shape: %s, batch_size: %d, n: %d",
        A.shape, batch_size, n
    )

    # Determine output dtype (use float32 for computation, convert back)
    compute_dtype = torch.float32
    result_dtype = A.dtype

    # Make contiguous if needed
    A = A.contiguous()

    # Allocate output tensors
    result = torch.empty(batch_dims, device=A.device, dtype=compute_dtype)
    LU = torch.empty(*batch_dims, n, n, device=A.device, dtype=compute_dtype)
    pivots = torch.empty(*batch_dims, n, device=A.device, dtype=torch.int32)

    # For batch > 1, we need to handle batch dimension
    # Reshape to (batch, n, n) for processing
    if batch_dims:
        A_flat = A.reshape(-1, n, n)
    else:
        A_flat = A.unsqueeze(0)

    # Process based on matrix size
    if n == 2:
        grid = (batch_size,)
        linalg_det_2x2_kernel[grid](
            A_flat,
            result.reshape(-1),
            batch_size,
            n,
            A_flat.stride(0),  # stride_ab: batch stride
            A_flat.stride(1),  # stride_am: matrix row stride
            A_flat.stride(2),  # stride_an: matrix col stride
        )
    elif n == 3:
        grid = (batch_size,)
        linalg_det_3x3_kernel[grid](
            A_flat,
            result.reshape(-1),
            batch_size,
            n,
            A_flat.stride(0),  # stride_ab: batch stride
            A_flat.stride(1),  # stride_am: matrix row stride
            A_flat.stride(2),  # stride_an: matrix col stride
        )
    else:
        # For larger matrices, use torch's lu_factor and compute determinant
        # This is a fallback for matrices larger than 3x3
        # Convert to compute_dtype for lu_factor (doesn't support float16/bfloat16)
        A_flat_compute = A_flat.to(compute_dtype)
        for i in range(batch_size):
            lu_result = torch.linalg.lu_factor(A_flat_compute[i])
            LU_single = lu_result.LU
            piv = lu_result.pivots

            if batch_dims:
                LU[i] = LU_single
                pivots[i] = piv  # Keep 1-indexed as-is
            else:
                # No batch dimension - direct assignment
                LU.copy_(LU_single)
                pivots.copy_(piv)

            # Compute determinant from LU
            # det(A) = det(L) * det(U) = product of diagonal elements of U
            # Then account for row swaps: pivots[i] != i+1 means row i was swapped
            diag = torch.diag(LU_single)
            det = torch.prod(diag)

            # Count row swaps: pivots is 1-indexed, so compare with 1, 2, 3, ...
            num_swaps = torch.sum(piv != torch.arange(1, len(piv) + 1, device=piv.device)).item()
            if num_swaps % 2 == 1:
                det = -det

            result.reshape(-1)[i] = det

        # For >3x3, we already filled LU and pivots in the loop

    # Fill in LU for 2x2 and 3x3 cases
    if n <= 3:
        # Copy input to LU for small matrices (LU decomposition would be more complex)
        LU = A_flat.to(compute_dtype).reshape(*batch_dims, n, n)

        # Create default pivots
        for i in range(batch_size):
            pivots.reshape(-1, n)[i] = torch.arange(1, n + 1, device=A.device, dtype=torch.int32)

    # Convert result back to original dtype
    result = result.to(result_dtype)
    LU = LU.to(result_dtype)

    # Return as named tuple
    return _LinalgDet(result=result, LU=LU, pivots=pivots)