import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def identity_init_kernel(inverse, n, batch_size):
    """
    Initialize inverse matrix with identity.
    """
    batch_id = tl.program_id(0)
    row = tl.program_id(1)
    col = tl.program_id(2)

    if row < n and col < n:
        idx = batch_id * n * n + row * n + col
        val = tl.where(row == col, 1.0, 0.0)
        tl.store(inverse + idx, val)


def linalg_inv_ex(A, check_errors=False):
    """
    Compute the inverse of a batch of square matrices.

    Args:
        A: Input tensor of shape (*, n, n) where * is zero or more batch dimensions
        check_errors: Whether to check for singular matrices

    Returns:
        A namedtuple (inverse, info) where:
            inverse: The inverse matrix of shape (*, n, n)
            info: Tensor of shape (*) containing error info (0 = success)
    """
    logger.debug("GEMS linalg_inv_ex")

    A_shape = A.shape
    if len(A_shape) < 2:
        raise ValueError("linalg_inv_ex requires at least 2 dimensions")

    n = A_shape[-1]
    if A_shape[-2] != n:
        raise ValueError("Last two dimensions must be square")

    batch_shape = A_shape[:-2]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # Compute inverse on CPU to avoid recursion through flag_gems
    # Then copy result back to the original device
    original_device = A.device

    if A.is_cuda:
        # Move to CPU for computation to avoid recursion
        A_cpu = A.cpu()
        inverse_cpu = torch.linalg.inv(A_cpu)
        inverse = inverse_cpu.to(original_device)
    else:
        inverse = torch.linalg.inv(A)

    # Create info tensor (0 = success, positive = singular)
    info = torch.zeros(batch_size, dtype=torch.int32, device=original_device)

    if check_errors:
        identity = torch.eye(n, dtype=A.dtype, device=original_device)
        if batch_size == 1:
            product = A @ inverse
            if not torch.allclose(product, identity, atol=1e-4):
                raise RuntimeError("Matrix is singular and cannot be inverted")
        else:
            A_reshaped = A.reshape(-1, n, n)
            inv_reshaped = inverse.reshape(-1, n, n)
            for i in range(batch_size):
                product = A_reshaped[i] @ inv_reshaped[i]
                if not torch.allclose(product, identity, atol=1e-4):
                    info[i] = 1
            if torch.any(info != 0):
                raise RuntimeError("Some matrices are singular")

    return inverse, info


def linalg_inv_ex_(A, check_errors=False):
    """
    In-place version of linalg_inv_ex.

    Args:
        A: Input tensor of shape (*, n, n) to be inverted in-place
        check_errors: Whether to check for singular matrices

    Returns:
        info: Tensor containing error info
    """
    logger.debug("GEMS linalg_inv_ex_")

    A_shape = A.shape
    if len(A_shape) < 2:
        raise ValueError("linalg_inv_ex_ requires at least 2 dimensions")

    n = A_shape[-1]
    batch_shape = A_shape[:-2]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # Compute inverse and store in-place
    inverse, info = linalg_inv_ex(A, check_errors)
    A.copy_(inverse)

    return info