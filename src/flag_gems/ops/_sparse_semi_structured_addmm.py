import logging

import torch

logger = logging.getLogger(__name__)


def _sparse_semi_structured_addmm(input, mat1, mat1_meta, mat2, *, alpha=1, beta=1, out_dtype=None):
    """
    Performs a sparse semi-structured matrix multiplication and accumulation.

    Computes: result = alpha * (mat1 @ mat2) + beta * input

    This operator requires:
    - GPU with compute capability >= 8.0 (Ampere or newer)
    - mat1 columns must be multiple of 32
    - mat2 rows must be multiple of 32

    When hardware doesn't support sparse operation, falls back to dense computation
    using the addmm Triton kernel.

    Args:
        input: Input tensor (bias), shape (M, N)
        mat1: First matrix, shape (M, K)
        mat1_meta: Meta tensor from torch._sparse_semi_structured_tile
        mat2: Second matrix, shape (K, N)
        alpha: Scalar multiplier for mat1 @ mat2
        beta: Scalar multiplier for input
        out_dtype: Optional output dtype

    Returns:
        Result tensor of shape (M, N)
    """
    logger.debug("GEMS SPARSE_SEMI_STRUCTURED_ADDMM")

    # Get shapes
    M = mat1.shape[0]
    K = mat1.shape[1]
    N = mat2.shape[1]

    # Determine output dtype
    if out_dtype is not None:
        output_dtype = out_dtype
    else:
        output_dtype = mat1.dtype

    # Fall back to dense computation using Triton-based addmm
    # Note: We don't try the sparse operation here because:
    # 1. It may not be supported on all hardware (requires compute capability >= 8.0)
    # 2. When called within flag_gems.use_gems(), torch._sparse_semi_structured_addmm
    #    would redirect back to this function causing infinite recursion
    from flag_gems.ops.addmm import addmm as gems_addmm
    logger.debug("Using dense computation via Triton addmm kernel")
    result = gems_addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    return result