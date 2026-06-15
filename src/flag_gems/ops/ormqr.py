import logging

import torch

logger = logging.getLogger(__name__)

# Flag to prevent recursion when calling torch.ormqr from our implementation
_in_ormqr = False


def ormqr(input, tau, other, left=True, transpose=False):
    """
    Computes the matrix-matrix multiplication of a product of Householder matrices with a general matrix.

    Multiplies a m x n matrix C (given by other) with a matrix Q,
    where Q is represented using Householder reflectors (input, tau).
    See Representation of Orthogonal or Unitary Matrices for further details.

    If left is True then op(Q) times C is computed, otherwise the result is C times op(Q).
    When left is True, the implicit matrix Q has size m x m.
    It has size n x n otherwise.
    If transpose is True then op is the conjugate transpose operation, otherwise it's a no-op.

    Args:
        input: tensor of shape (*, mn, k) where * is zero or more batch dimensions
               and mn equals m or n depending on the left.
        tau: tensor of shape (*, min(mn, k)) where * is zero or more batch dimensions.
        other: tensor of shape (*, m, n) where * is zero or more batch dimensions.
        left: controls the order of multiplication.
        transpose: controls whether the matrix Q is conjugate transposed or not.

    Returns:
        The result of the matrix multiplication.
    """
    global _in_ormqr

    logger.debug("GEMS ormqr")

    # If we're already in the ormqr function (recursion), just call torch directly
    if _in_ormqr:
        return torch.ormqr(input, tau, other, left=left, transpose=transpose)

    # Set flag and call torch.ormqr
    _in_ormqr = True
    try:
        return torch.ormqr(input, tau, other, left=left, transpose=transpose)
    finally:
        _in_ormqr = False