import logging

import torch

import flag_gems
from flag_gems.ops.attention import flash_attn_varlen_func

logger = logging.getLogger(__name__)


def page_attention(
    query,
    key_cache,
    value_cache,
    block_table,
    context_lengths,
    max_context_length,
    scale=None,
    is_causal=False,
):
    """Page attention operator for efficient KV cache management.

    This operator implements attention using paged KV cache, which is commonly
    used in LLM inference systems like vLLM.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seqlen_q, head_dim)
        key_cache: Paged key cache of shape (num_pages, block_size, num_kv_heads, head_dim)
        value_cache: Paged value cache of shape (num_pages, block_size, num_kv_heads, head_dim)
        block_table: Block table of shape (batch_size, max_blocks_per_seq)
        context_lengths: Actual sequence lengths of shape (batch_size,)
        max_context_length: Maximum context length
        scale: Optional scale factor for attention scores
        is_causal: Whether to apply causal masking

    Returns:
        Attention output of shape (batch_size, num_heads, seqlen_q, head_dim)
    """
    logger.debug("GEMS PAGE_ATTENTION")

    # Input validation
    assert query.dim() == 4, f"query must be 4D, got {query.dim()}D"
    assert key_cache.dim() == 4, f"key_cache must be 4D, got {key_cache.dim()}D"
    assert value_cache.dim() == 4, f"value_cache must be 4D, got {value_cache.dim()}D"
    assert block_table.dim() == 2, f"block_table must be 2D, got {block_table.dim()}D"
    assert context_lengths.dim() == 1, f"context_lengths must be 1D, got {context_lengths.dim()}D"

    batch_size, num_heads, seqlen_q, head_dim = query.shape
    num_pages, block_size, num_kv_heads, _ = key_cache.shape

    assert key_cache.shape == value_cache.shape, "key_cache and value_cache must have same shape"
    assert block_table.shape[0] == batch_size, "block_table batch size must match query batch size"
    assert context_lengths.shape[0] == batch_size, "context_lengths batch size must match query batch size"

    # Support for GQA (grouped query attention)
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    head_dim_k = key_cache.shape[3]
    assert head_dim == head_dim_k, f"query head_dim ({head_dim}) must match key_cache head_dim ({head_dim_k})"

    # Prepare inputs for flash_attn_varlen_func
    # The flash_attn_varlen_func expects:
    # - q: (total_q, num_heads, head_dim) - flattened across batch
    # - k, v: (num_pages, block_size, num_kv_heads, head_dim) for paged
    # - cu_seqlens_q, cu_seqlens_k: cumulative sequence lengths

    # For page attention, we treat each query position independently
    # since the block_table maps each position to the correct page

    # Create cumulative sequence lengths
    # Each batch element has seqlen_q tokens
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=query.device)
    cu_seqlens_q[1:] = torch.cumsum(torch.tensor([seqlen_q] * batch_size, device=query.device), dim=0)
    total_q = batch_size * seqlen_q

    # For paged attention with block_table, we need seqused_k
    # This indicates how many tokens are used for each sequence in the KV cache
    # seqused_k = context_length (the actual number of tokens, not blocks)
    # Ensure it's on the correct device
    seqused_k = context_lengths.to(device=query.device, dtype=torch.int32)

    # Reshape query from (batch, heads, seqlen, head) to (total_q, heads, head)
    query = query.transpose(1, 2).reshape(total_q, num_heads, head_dim)

    # Determine max seqlen values
    max_seqlen_q = seqlen_q
    max_seqlen_k = max_context_length

    # Handle scale
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # Prepare block table
    # block_table is (batch, max_blocks_per_seq), need to ensure it's int32
    block_table = block_table.to(torch.int32)

    # Call flash_attn_varlen_func with paged KV cache
    # When using block_table, we use seqused_k instead of cu_seqlens_k
    out = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=None,  # Use seqused_k instead when using block_table
        seqused_k=seqused_k,
        q_v=None,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=is_causal,
        window_size=None,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=block_table,
        return_softmax_lse=False,
        out=None,
    )

    # Reshape output from (total_q, num_heads, head_dim) to (batch, num_heads, seqlen_q, head_dim)
    out = out.reshape(batch_size, seqlen_q, num_heads, head_dim)
    out = out.transpose(1, 2)

    return out