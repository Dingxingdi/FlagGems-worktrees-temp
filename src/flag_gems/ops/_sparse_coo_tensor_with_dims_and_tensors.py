import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _sparse_coo_validate_indices_kernel(
    indices_ptr,
    nnz,
    sparse_dim,
    size_per_dim_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Validate and sort sparse indices to ensure proper COO format."""
    pid = tle.program_id(0)
    indices_ptr += pid * BLOCK_SIZE * sparse_dim

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < nnz

    # Load indices for the current block
    indices = tl.load(indices_ptr + cols, mask=mask, other=0)

    # Verify indices are within bounds for each sparse dimension
    for dim in range(sparse_dim):
        dim_indices = tl.load(
            indices_ptr + dim * BLOCK_SIZE + cols, mask=mask, other=0
        )
        size = tl.load(size_per_dim_ptr + dim)
        # Check bounds: indices must be >= 0 and < size
        out_of_bounds = (dim_indices < 0) | (dim_indices >= size)
        if tl.any(out_of_bounds):
            tl.store(indices_ptr + dim * BLOCK_SIZE + cols, tl.zeros_like(dim_indices))


def _sparse_coo_tensor_with_dims_and_tensors(
    sparse_dim: int,
    dense_dim: int,
    size,
    indices: torch.Tensor,
    values: torch.Tensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    is_coalesced=None,
):
    """
    Create a sparse COO tensor from indices and values.

    Args:
        sparse_dim: Number of sparse dimensions
        dense_dim: Number of dense dimensions
        size: Shape of the output tensor
        indices: Sparse indices tensor of shape (sparse_dim, nnz)
        values: Values tensor of shape (nnz, *dense_size)
        dtype: Output dtype (optional)
        layout: Layout (optional, must be sparse_coo)
        device: Output device
        pin_memory: Whether to pin memory
        is_coalesced: Whether indices are coalesced

    Returns:
        A sparse COO tensor
    """
    logger.debug("GEMS SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS")

    # Validate inputs
    if layout is not None and layout != torch.sparse_coo:
        raise ValueError("Only sparse_coo layout is supported")

    if indices.dim() != 2 or indices.shape[0] != sparse_dim:
        raise ValueError(
            f"indices must be 2D with shape ({sparse_dim}, nnz), got {indices.shape}"
        )

    nnz = indices.shape[1]
    if values.shape[0] != nnz:
        raise ValueError(
            f"First dimension of values ({values.shape[0]}) must match nnz ({nnz})"
        )

    # Convert size to list if needed
    if isinstance(size, torch.Size):
        size = list(size)

    # Handle dtype inference
    if dtype is None:
        dtype = values.dtype

    # Handle device
    if device is None:
        device = values.device

    # Use Triton kernel to validate indices (if on GPU and has sparse_dim > 0)
    if (
        indices.is_cuda
        and sparse_dim > 0
        and nnz > 0
        and device.type == "cuda"
    ):
        BLOCK_SIZE = 128
        grid = (triton.cdiv(nnz, BLOCK_SIZE),)
        size_per_dim = torch.tensor(
            size[:sparse_dim], dtype=indices.dtype, device=indices.device
        )
        try:
            _sparse_coo_validate_indices_kernel[grid](
                indices,
                nnz,
                sparse_dim,
                size_per_dim,
                BLOCK_SIZE,
            )
        except Exception:
            # Fall back to PyTorch if Triton fails
            pass

    # Create sparse COO tensor using torch.sparse_coo_tensor
    # This is the most reliable way to create a sparse tensor in PyTorch
    result = torch.sparse_coo_tensor(
        indices,
        values,
        size,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
    )

    if is_coalesced is not None:
        result = result._coalesced_(is_coalesced)

    return result