import logging

import torch

logger = logging.getLogger(__name__)


# Matrix exponential using Taylor series with scaling and squaring
# e^A = I + A + A^2/2! + A^3/3! + ...
# Uses scaling and squaring for numerical stability: e^A = (e^(A/2^s))^2^s


def compute_matrix_exp_2d(A):
    """
    Compute matrix exponential for a 2D matrix (N x N).
    Uses float32 for intermediate computations for numerical stability.
    """
    n = A.shape[0]
    original_dtype = A.dtype

    # Use float32 for computation regardless of input dtype
    # This provides a good balance between precision and compatibility
    if original_dtype == torch.float32:
        A_compute = A
    else:
        A_compute = A.to(torch.float32)
    A_compute = A_compute.contiguous()

    # Compute matrix norm for scaling (use float32 version)
    A_norm = torch.linalg.matrix_norm(A_compute, ord=1)

    # Determine scale factor based on matrix norm
    if A_norm > 1:
        max_scale = int(torch.ceil(torch.log2(A_norm)).item())
    else:
        max_scale = 0
    max_scale = max(0, max_scale)

    # Scale down the matrix
    if max_scale > 0:
        A_scaled = A_compute / (2.0 ** max_scale)
    else:
        A_scaled = A_compute

    # Taylor series: result = I + A + A^2/2! + A^3/3! + ...
    result = torch.eye(n, dtype=torch.float32, device=A.device)
    term = torch.eye(n, dtype=torch.float32, device=A.device)

    factorial = 1
    A_power = A_scaled.clone()

    # Taylor series iteration - use fixed number of iterations
    max_iter = 15
    for i in range(1, max_iter + 1):
        factorial = factorial * i
        term = A_power / factorial
        result = result + term
        A_power = torch.mm(A_power, A_scaled)

    # Squaring step: (e^(A/s))^2^s
    for _ in range(max_scale):
        result = torch.mm(result, result)

    # Convert back to original dtype
    return result.to(original_dtype)


def linalg_matrix_exp(A):
    """
    Compute matrix exponential.

    This implementation uses Taylor series with scaling and squaring
    for numerical stability.

    Args:
        A: Input tensor of shape (*, N, N)

    Returns:
        Matrix exponential of shape (*, N, N)
    """
    logger.debug("GEMS LINALG_MATRIX_EXP")

    # Ensure input is a square matrix
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Input must be a square matrix, got shape {A.shape}")

    if A.dim() < 2:
        raise ValueError(f"Input must be at least 2D, got {A.dim()}D tensor")

    # Handle batch dimensions
    if A.dim() == 2:
        # 2D matrix (N x N)
        return compute_matrix_exp_2d(A)
    else:
        # Batched matrix (*, N, N)
        batch_shape = A.shape[:-2]
        n = A.shape[-1]
        batch_size = 1
        for dim in batch_shape:
            batch_size *= dim

        # Process each matrix in the batch
        A_flat = A.view(batch_size, n, n)
        results = []
        for i in range(batch_size):
            result = compute_matrix_exp_2d(A_flat[i])
            results.append(result)

        return torch.stack(results).view(*batch_shape, n, n)


def linalg_matrix_exp_(A):
    """
    In-place matrix exponential.
    Note: Due to Triton compatibility, this returns a new tensor.
    """
    logger.debug("GEMS LINALG_MATRIX_EXP_")
    return linalg_matrix_exp(A)