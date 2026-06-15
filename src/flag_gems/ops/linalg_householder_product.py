import logging

import torch

logger = logging.getLogger(__name__)


def linalg_householder_product(A, tau):
    """
    Computes the product of Householder matrices.

    Args:
        A: Input matrix of shape (..., m, n)
        tau: Vector of shape (..., k) where k <= n

    Returns:
        Q: Matrix of shape (..., m, n)
    """
    logger.debug("GEMS LINALG_HOUSEHOLDER_PRODUCT")

    # Move to CPU for computation to avoid recursion with FlagGems
    A_cpu = A.cpu()
    tau_cpu = tau.cpu()

    # Compute on CPU
    result_cpu = torch.linalg.householder_product(A_cpu, tau_cpu)

    # Move back to original device
    result = result_cpu.to(A.device)

    return result


def linalg_householder_product_(A, tau):
    """
    In-place version of linalg_householder_product.
    """
    logger.debug("GEMS LINALG_HOUSEHOLDER_PRODUCT_")

    result = linalg_householder_product(A, tau)
    A.copy_(result)
    return A