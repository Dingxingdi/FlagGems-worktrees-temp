import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


def is_non_overlapping_and_dense(a: torch.Tensor) -> torch.Tensor:
    """
    Check if tensor's storage is non-overlapping and dense.

    A tensor is non-overlapping and dense when there exists a permutation of
    its dimensions that is contiguous.

    Implementation based on PyTorch's torch._prims_common.is_non_overlapping_and_dense.
    """
    logger.debug("GEMS IS_NON_OVERLAPPING_AND_DENSE")

    # Get shape and stride
    shape = a.shape
    ndim = len(shape)

    # Handle scalar tensors (0D)
    if ndim == 0:
        return True

    # For 1D tensors: non-overlapping and dense if stride[0] == 1
    if ndim == 1:
        return a.stride()[0] == 1

    # Get strides
    strides = a.stride()

    # Quick check: if the tensor is contiguous in row-major order,
    # stride[i] should equal product of shape[j] for j > i
    # Check if already contiguous
    expected_stride = 1
    is_contig = True
    for i in range(ndim - 1, -1, -1):
        if shape[i] > 1:
            if strides[i] != expected_stride:
                is_contig = False
                break
            expected_stride *= shape[i]
    if is_contig:
        return True

    # General case: check if there exists a permutation of dimensions
    # that would make the tensor contiguous
    #
    # Algorithm:
    # 1. Create (size, stride) pairs for each dimension
    # 2. Sort by stride (ascending)
    # 3. Check if the sorted tensor would be contiguous:
    #    - Expected stride starts at 1
    #    - For each (length, stride) pair:
    #      - If length == 1, skip
    #      - If stride != expected_stride, return False
    #      - Otherwise, expected_stride *= length

    # Create pairs and sort by stride
    lengths_and_strides = sorted(
        [(shape[i], strides[i]) for i in range(ndim)],
        key=lambda x: x[1]
    )

    # Check if the sorted tensor would be contiguous
    expected_stride = 1
    for length, stride in lengths_and_strides:
        if length == 1:
            continue
        if stride != expected_stride:
            return False
        expected_stride *= length

    return True


def is_non_overlapping_and_dense_(A: torch.Tensor) -> torch.Tensor:
    """In-place version - not applicable for this operator."""
    raise NotImplementedError(
        "is_non_overlapping_and_dense_ is not supported as this operator "
        "does not modify the tensor, it only queries its properties."
    )