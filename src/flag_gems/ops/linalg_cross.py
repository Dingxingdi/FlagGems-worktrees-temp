import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def linalg_cross_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    num_vectors: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cross product kernel for 3D vectors.

    Computes a x b for each 3D vector pair.
    For each vector pair (a, b):
        result[0] = a[1] * b[2] - a[2] * b[1]
        result[1] = a[2] * b[0] - a[0] * b[2]
        result[2] = a[0] * b[1] - a[1] * b[0]
    """
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_vectors

    # Each thread loads one 3-element vector from each input
    # a_ptr and b_ptr point to flattened (num_vectors, 3) tensors
    a0 = tl.load(a_ptr + offsets * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    a1 = tl.load(a_ptr + offsets * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    a2 = tl.load(a_ptr + offsets * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    b0 = tl.load(b_ptr + offsets * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + offsets * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(b_ptr + offsets * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    # Compute cross product components
    r0 = a1 * b2 - a2 * b1
    r1 = a2 * b0 - a0 * b2
    r2 = a0 * b1 - a1 * b0

    # Store results
    tl.store(out_ptr + offsets * 3 + 0, r0, mask=mask)
    tl.store(out_ptr + offsets * 3 + 1, r1, mask=mask)
    tl.store(out_ptr + offsets * 3 + 2, r2, mask=mask)


def linalg_cross(a, b, *, dim=-1):
    """Compute cross product of two tensors along the given dimension.

    Args:
        a: First input tensor
        b: Second input tensor
        dim: Dimension along which to compute cross product (default: -1)

    Returns:
        Cross product tensor
    """
    logger.debug("GEMS LINALG_CROSS")

    # Input validation
    assert a.shape == b.shape, "Input tensors must have the same shape"

    # Normalize dim
    dim = dim if dim >= 0 else a.dim() + dim

    # Check that the cross product dimension has size 3
    cross_dim_size = a.shape[dim]
    if cross_dim_size != 3:
        raise ValueError(
            f"linalg.cross: cross product dimension must have size 3, got {cross_dim_size}"
        )

    # If dim is not the last dimension, we need to permute to make it last
    if dim != a.dim() - 1:
        # Build permutation: move dim to last position
        perm = list(range(a.dim()))
        perm.remove(dim)
        perm.append(dim)
        a = a.permute(perm)
        b = b.permute(perm)

    # Now the cross dimension is at position -1 (last)
    # Get prefix shape (all dimensions except the last)
    prefix_shape = list(a.shape[:-1])
    num_vectors = 1
    for s in prefix_shape:
        num_vectors *= s

    # Flatten inputs to (num_vectors, 3)
    a_flat = a.reshape(num_vectors, 3).contiguous()
    b_flat = b.reshape(num_vectors, 3).contiguous()

    # Allocate output
    out_flat = torch.empty((num_vectors, 3), dtype=torch.float32, device=a.device)

    # Define grid
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_vectors, BLOCK_SIZE),)

    # Launch kernel
    linalg_cross_kernel[grid](
        a_flat, b_flat, out_flat, num_vectors, BLOCK_SIZE
    )

    # Reshape output back to original shape (with dim moved back if needed)
    out_shape = a.shape
    out = out_flat.reshape(out_shape).to(a.dtype)

    # If we permuted, permute back
    if dim != a.dim() - 1:
        # We need to inverse the permutation
        # Original perm moved dim to end, so inverse moves last to dim
        inv_perm = list(range(a.dim()))
        inv_perm.remove(a.dim() - 1)
        inv_perm.insert(dim, a.dim() - 1)
        out = out.permute(inv_perm)

    return out


def linalg_cross_(a, b, *, dim=-1):
    """In-place cross product (not supported, raises error)."""
    raise NotImplementedError("linalg_cross_ (in-place) is not supported")