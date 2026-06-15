import logging

import torch

logger = logging.getLogger(__name__)


def _cslt_sparse_mm(
    compressed_A: torch.Tensor,
    dense_B: torch.Tensor,
    bias: torch.Tensor = None,
    alpha: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    transpose_result: bool = False,
    alg_id: int = 0,
    split_k: int = 1,
    split_k_one_kernel: bool = True,
):
    """
    Compressed sparse matrix multiplication using cuSPARSELt.

    This operator performs: Y = alpha * (A @ B) + bias
    where A is a structured sparse matrix in compressed format and B is dense.

    Args:
        compressed_A: Compressed sparse tensor A
        dense_B: Dense tensor B
        bias: Optional bias tensor to add
        alpha: Optional scalar multiplier
        out_dtype: Output data type
        transpose_result: Whether to transpose the result
        alg_id: Algorithm ID for cuSPARSELt
        split_k: Split K factor
        split_k_one_kernel: Whether to use one kernel for split K

    Returns:
        Result tensor
    """
    logger.debug("GEMS _CSLT_SPARSE_MM")
    return torch._cslt_sparse_mm(
        compressed_A,
        dense_B,
        bias=bias,
        alpha=alpha,
        out_dtype=out_dtype,
        transpose_result=transpose_result,
        alg_id=alg_id,
        split_k=split_k,
        split_k_one_kernel=split_k_one_kernel,
    )