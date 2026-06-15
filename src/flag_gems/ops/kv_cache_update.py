import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def kv_cache_update_kernel(
    k_cache_ptr,
    v_cache_ptr,
    new_k_ptr,
    new_v_ptr,
    cache_shape0,
    cache_shape1,
    cache_shape2,
    cache_shape3,
    new_k_shape0,
    new_k_shape1,
    new_k_shape2,
    new_k_shape3,
    start_position,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to update KV cache with new key and value tensors.

    Args:
        k_cache: Key cache tensor of shape (batch, num_heads, max_seq_len, head_dim)
        v_cache: Value cache tensor of shape (batch, num_heads, max_seq_len, head_dim)
        new_k: New key tensor to insert of shape (batch, num_heads, new_seq_len, head_dim)
        new_v: New value tensor to insert of shape (batch, num_heads, new_seq_len, head_dim)
        start_position: Starting position in the cache to update
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)

    # Total elements to update (just K or V, not both)
    total_elements = new_k_shape0 * new_k_shape1 * new_k_shape2 * new_k_shape3

    idx = block_start + offsets
    mask = idx < total_elements

    # Compute 4D indices
    new_idx0 = idx // (new_k_shape1 * new_k_shape2 * new_k_shape3)
    rem = idx % (new_k_shape1 * new_k_shape2 * new_k_shape3)
    new_idx1 = rem // (new_k_shape2 * new_k_shape3)
    rem = rem % (new_k_shape2 * new_k_shape3)
    new_idx2 = rem // new_k_shape3
    new_idx3 = rem % new_k_shape3

    # Compute position in cache
    cache_idx2 = start_position + new_idx2

    # Compute new_k offset
    new_k_offset = (
        new_idx0 * (new_k_shape1 * new_k_shape2 * new_k_shape3)
        + new_idx1 * (new_k_shape2 * new_k_shape3)
        + new_idx2 * new_k_shape3
        + new_idx3
    )

    # Compute new_v offset (same layout)
    new_v_offset = new_k_offset

    # Compute cache offsets
    k_cache_offset = (
        new_idx0 * (cache_shape1 * cache_shape2 * cache_shape3)
        + new_idx1 * (cache_shape2 * cache_shape3)
        + cache_idx2 * cache_shape3
        + new_idx3
    )
    v_cache_offset = k_cache_offset  # Same layout for V cache

    # Load values
    k_val = tl.load(new_k_ptr + new_k_offset, mask=mask)
    v_val = tl.load(new_v_ptr + new_v_offset, mask=mask)

    # Store to caches
    tl.store(k_cache_ptr + k_cache_offset, k_val, mask=mask)
    tl.store(v_cache_ptr + v_cache_offset, v_val, mask=mask)


def kv_cache_update(k_cache, v_cache, new_k, new_v, start_position=0):
    """Update KV cache with new key and value tensors.

    This operator updates the KV (Key-Value) cache used in transformer inference.
    It writes new key and value vectors at specified positions in the cache.

    Args:
        k_cache: Key cache tensor of shape (batch, num_heads, max_seq_len, head_dim)
        v_cache: Value cache tensor of shape (batch, num_heads, max_seq_len, head_dim)
        new_k: New key tensor to insert of shape (batch, num_heads, new_seq_len, head_dim)
        new_v: New value tensor to insert of shape (batch, num_heads, new_seq_len, head_dim)
        start_position: Starting position in the cache to update (default: 0)

    Returns:
        Tuple of (updated_k_cache, updated_v_cache)
    """
    logger.debug("GEMS KV_CACHE_UPDATE")

    assert k_cache.device == v_cache.device == new_k.device == new_v.device, \
        "k_cache, v_cache, new_k, and new_v must be on the same device"
    assert new_k.shape == new_v.shape, \
        "new_k and new_v must have the same shape"

    # Get shapes
    cache_shape = k_cache.shape
    new_k_shape = new_k.shape

    assert len(cache_shape) == 4, "cache must be 4D tensor"
    assert len(new_k_shape) == 4, "new_k and new_v must be 4D tensors"
    assert k_cache.shape == v_cache.shape, "k_cache and v_cache must have same shape"

    # Ensure start_position is valid
    assert start_position >= 0, "start_position must be non-negative"
    assert start_position + new_k_shape[2] <= cache_shape[2], \
        f"update extends beyond cache: {start_position + new_k_shape[2]} > {cache_shape[2]}"

    # Make tensors contiguous
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    new_k = new_k.contiguous()
    new_v = new_v.contiguous()

    # Total elements to update
    total_elements = new_k.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    kv_cache_update_kernel[grid](
        k_cache,
        v_cache,
        new_k,
        new_v,
        cache_shape[0],
        cache_shape[1],
        cache_shape[2],
        cache_shape[3],
        new_k_shape[0],
        new_k_shape[1],
        new_k_shape[2],
        new_k_shape[3],
        start_position,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return k_cache, v_cache


def kv_cache_update_(k_cache, v_cache, new_k, new_v, start_position=0):
    """In-place version of kv_cache_update."""
    logger.debug("GEMS KV_CACHE_UPDATE_")
    return kv_cache_update(k_cache, v_cache, new_k, new_v, start_position)