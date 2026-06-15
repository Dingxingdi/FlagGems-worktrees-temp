import logging
from typing import List, Optional

import torch

from flag_gems.ops.attention import flash_attn_varlen_func

logger = logging.getLogger(__name__)


def Paged_Attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Optional[tuple] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
):
    """Paged attention operator for FlagGems.

    This operator implements attention with paged key/value caching,
    commonly used in LLM inference.

    Args:
        query: Query tensor of shape (total_q, num_query_heads, head_dim)
        key_cache: Paged key cache of shape (num_blocks, block_size, num_kv_heads, head_dim)
        value_cache: Paged value cache of shape (num_blocks, block_size, num_kv_heads, head_dim)
        query_lens: List of query lengths for each sequence
        kv_lens: List of kv lengths for each sequence
        block_tables: Block table mapping of shape (num_seqs, max_num_blocks)
        softmax_scale: Attention scale factor. If None, defaults to 1/sqrt(head_dim)
        causal: Whether to apply causal masking. Default True.
        window_size: Optional tuple of (left, right) for sliding window attention
        softcap: Soft capping value for attention logits. Default 0.0 (no capping)
        alibi_slopes: Optional ALiBi slopes of shape (num_query_heads,) or (batch_size, num_query_heads)
        block_size: Block size (derived from key_cache if not provided)

    Returns:
        Output tensor of shape (total_q, num_query_heads, head_dim)
    """
    logger.debug("GEMS PAGED_ATTENTION")

    num_seqs = len(query_lens)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = query.shape[2]

    # Get block_size from key_cache if not provided
    if block_size is None:
        block_size = key_cache.shape[1]

    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    # Prepare cumulative sequence lengths for queries
    cu_query_lens = torch.zeros(
        num_seqs + 1, dtype=torch.int32, device=query.device
    )
    cu_query_lens[1:] = torch.cumsum(
        torch.tensor(query_lens, device=query.device, dtype=torch.int32), dim=0
    )

    # Prepare sequence lengths used for KV
    seqused_k = torch.tensor(kv_lens, device=key_cache.device, dtype=torch.int32)

    # Set window size
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = window_size

    # Call flash_attn_varlen_func with block_table
    output = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=real_window_size,
        block_table=block_tables,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        fa_version=2,
    )

    return output