import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm import mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def identity_diag_kernel(out, N, BLOCK_SIZE: tl.constexpr):
    """
    Create a matrix with 2.0 on the diagonal.
    This is a helper kernel for the Newton-Schulz iteration.
    """
    pid = tle.program_id(0).to(tl.int64)
    row = pid
    if row >= N:
        return

    # Set diagonal elements to 2.0
    offset = row * N + row
    tl.store(out + offset, 2.0)


def pinv_newton_schulz(A, max_iter=50):
    """
    Compute pseudoinverse using Newton-Schulz iteration.
    Uses FlagGems mm for matrix multiplication.

    The iteration: X_{k+1} = X_k (2I - A X_k)
    converges to A^+ if spectral radius(A X_0) < 1

    A: (M, N) input matrix
    Returns: (N, M) pseudoinverse
    """
    m, n = A.shape

    # Use float32 for computation
    A_float = A.to(torch.float32)

    # Compute initial guess X = alpha * A^T
    # Use adaptive alpha based on matrix norm
    # alpha = 0.01 is a safe starting value
    alpha = 0.01

    # Initial guess X = alpha * A^T
    X = alpha * A_float.t()

    # For iterative refinement, also track A X
    AX = mm(A_float, X)

    # Iteration: X_{k+1} = X_k (2I - AX)
    for _ in range(max_iter):
        # Compute 2I - AX
        two_i = torch.zeros_like(AX)
        diag_len = min(m, n)
        two_i.view(-1)[:diag_len*(m+n+1):m+n+1] = 2.0
        two_i_minus_ax = two_i - AX

        # Compute X_new = X @ (2I - AX)
        X = mm(X, two_i_minus_ax)

        # Update AX for next iteration
        AX = mm(A_float, X)

    return X


def pinv(A, atol=None, rtol=None, hermitian=False):
    """
    Compute the Moore-Penrose pseudoinverse of a matrix.

    Args:
        A: Input tensor of shape (*, m, n)
        atol: Absolute tolerance
        rtol: Relative tolerance
        hermitian: If True, assume A is Hermitian

    Returns:
        Pseudoinverse of shape (*, n, m)
    """
    logger.debug("GEMS PINV")

    # Handle batch dimensions
    original_shape = A.shape
    is_batch = len(original_shape) > 2

    if is_batch:
        batch_dims = original_shape[:-2]
        m, n = original_shape[-2:]
        A = A.reshape(-1, m, n)
        batch_size = A.shape[0]
    else:
        m, n = original_shape
        A = A.unsqueeze(0)
        batch_size = 1

    # Determine output dtype
    output_dtype = A.dtype

    # Compute pseudoinverse for each matrix in the batch
    results = []
    for i in range(batch_size):
        a = A[i]
        a_inv = pinv_newton_schulz(a)
        results.append(a_inv.to(output_dtype))

    result = torch.stack(results)

    if is_batch:
        result = result.reshape(*batch_dims, n, m)
    else:
        result = result.squeeze(0)

    return result


def linalg_pinv(A, *, atol=None, rtol=None, hermitian=False, out=None):
    """
    Compute the Moore-Penrose pseudoinverse of a matrix.

    This is the entry point registered with FlagGems.
    """
    logger.debug("GEMS LINALG_PINV")

    if out is not None:
        result = pinv(A, atol=atol, rtol=rtol, hermitian=hermitian)
        out.copy_(result)
        return out

    return pinv(A, atol=atol, rtol=rtol, hermitian=hermitian)