import logging
from typing import List

import torch

from flag_gems.ops.mm import mm

logger = logging.getLogger(__name__)


def _compute_optimal_order(tensors: List[torch.Tensor]) -> List[int]:
    """
    Compute the optimal order of matrix multiplications using dynamic programming.
    Returns a list of indices representing the optimal order.
    """
    n = len(tensors)
    if n <= 2:
        return list(range(n))

    # dimensions[i] gives the output dimension of tensor i
    # For matrices A (m x n), B (n x p), C (p x q):
    # dims = [m, n, p, q] where A.shape = (m, n), B.shape = (n, p), C.shape = (p, q)
    dims = [tensors[i].shape[0] for i in range(n)]
    dims.append(tensors[-1].shape[1])

    # dp[i][j] = minimum cost to multiply matrices i through j
    # cost of multiplying (i..k) with (k+1..j) is dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
    dp = [[0] * n for _ in range(n)]
    # split[i][j] = optimal split point k for matrices i through j
    split = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    # Reconstruct the optimal order using the split table
    def reconstruct(i: int, j: int) -> List[int]:
        if i == j:
            return [i]
        k = split[i][j]
        left = reconstruct(i, k)
        right = reconstruct(k + 1, j)
        return left + right

    return reconstruct(0, n - 1)


def _multiply_ordered(tensors: List[torch.Tensor], order: List[int]) -> torch.Tensor:
    """Multiply tensors according to the given order."""
    if len(order) == 1:
        return tensors[order[0]]

    result = tensors[order[0]]
    for i in range(1, len(order)):
        result = mm(result, tensors[order[i]])
    return result


def multi_dot(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Efficiently multiplies two or more matrices by reordering the multiplications
    so that the fewest arithmetic operations are performed.
    """
    logger.debug("GEMS MULTI_DOT")
    if len(tensors) < 2:
        raise RuntimeError("multi_dot expects at least 2 tensors")

    # Handle the case with exactly 2 tensors - just do matrix multiplication
    if len(tensors) == 2:
        return mm(tensors[0], tensors[1])

    # Validate dimensions
    for i in range(len(tensors) - 1):
        if tensors[i].shape[-1] != tensors[i + 1].shape[0]:
            raise RuntimeError(
                f"Incompatible dimensions for matrix multiplication: "
                f"{tensors[i].shape} @ {tensors[i + 1].shape}"
            )

    # Compute optimal order and multiply
    order = _compute_optimal_order(tensors)
    return _multiply_ordered(tensors, order)