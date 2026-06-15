import logging

import torch

from flag_gems.ops.bmm import bmm
from flag_gems.ops.mm import mm

logger = logging.getLogger(__name__)


def linalg_matrix_power(A: torch.Tensor, n: int) -> torch.Tensor:
    """
    Compute the n-th power of a square matrix.

    Args:
        A: Input tensor of shape (*, m, m) where * is zero or more batch dimensions.
        n: Integer exponent.

    Returns:
        Tensor of shape (*, m, m) - the n-th power of the input matrix.
    """
    logger.debug("GEMS linalg_matrix_power")

    # Validate input
    if A.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("Input matrix must be square")

    # Handle n=0 case - return identity matrix
    if n == 0:
        m = A.shape[-1]
        batch_shape = A.shape[:-2]
        dtype = A.dtype
        device = A.device

        # Create identity matrix with proper batch dimensions
        identity = torch.eye(m, m, dtype=dtype, device=device)
        for dim_size in batch_shape:
            identity = identity.unsqueeze(0).expand(dim_size, *identity.shape)
        return identity

    # Store original dtype for final conversion
    original_dtype = A.dtype
    needs_dtype_conversion = A.dtype in (torch.float16, torch.bfloat16)

    # For low precision dtypes, we need to convert to float32 for numerical stability
    # especially when computing inverse
    if needs_dtype_conversion:
        A = A.to(torch.float32)

    # Handle negative n - compute inverse first
    if n < 0:
        A = torch.linalg.inv(A)
        n = -n

    # Determine if we should use bmm or mm based on whether there's a batch dimension
    is_batched = A.ndim > 2
    matmul = bmm if is_batched else mm

    # Use binary exponentiation for positive n
    result = None
    base = A
    current_n = n

    while current_n > 0:
        if current_n % 2 == 1:
            if result is None:
                result = base
            else:
                result = matmul(result, base)
        current_n //= 2
        if current_n > 0:
            base = matmul(base, base)

    # Convert back to original dtype if needed
    if needs_dtype_conversion:
        result = result.to(original_dtype)

    return result


def linalg_matrix_power_(A: torch.Tensor, n: int) -> torch.Tensor:
    """
    In-place version of linalg_matrix_power. Note: Since matrix power requires
    creating intermediate results, this is not truly in-place.
    """
    logger.debug("GEMS linalg_matrix_power_")
    result = linalg_matrix_power(A, n)
    A.copy_(result)
    return A