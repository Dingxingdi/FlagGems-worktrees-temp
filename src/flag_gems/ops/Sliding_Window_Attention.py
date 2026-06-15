import logging

import torch

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def sliding_window_attention(
    query,
    key,
    value,
    window_size,
    scale=None,
):
    """
    Sliding Window Attention operator.

    Each query position attends only to keys within a fixed window size.

    Args:
        query: (batch, num_heads, seq_len, head_dim)
        key: (batch, num_heads, seq_len, head_dim)
        value: (batch, num_heads, seq_len, head_dim)
        window_size: int, the window size for sliding window attention
        scale: optional scale factor for attention scores

    Returns:
        output: (batch, num_heads, seq_len, head_dim)
    """
    logger.debug("GEMS SLIDING WINDOW ATTENTION")

    # Validate inputs
    assert query.shape[-1] == key.shape[-1] == value.shape[-1]
    assert query.shape[1] == key.shape[1] == value.shape[1]
    assert query.shape[0] == key.shape[0] == value.shape[0]

    # Get dimensions
    batch, num_heads, seq_len, head_dim = query.shape

    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    # Import flash attention - it supports window_size
    from flag_gems.ops.attention import flash_attn_varlen_func

    # Flash attention expects (total_q, nheads, headdim) format
    # where total_q = batch * seq_len
    # Convert from (batch, num_heads, seq_len, head_dim) to (batch*seq_len, num_heads, head_dim)
    q = query.transpose(1, 2).reshape(batch * seq_len, num_heads, head_dim)
    k = key.transpose(1, 2).reshape(batch * seq_len, num_heads, head_dim)
    v = value.transpose(1, 2).reshape(batch * seq_len, num_heads, head_dim)

    # Create cumulative sequence lengths for flash attention
    # For uniform sequence lengths, cu_seqlens = [0, seq_len, 2*seq_len, ..., batch*seq_len]
    cu_seqlens = torch.arange(
        0, (batch + 1) * seq_len, step=seq_len, device=query.device, dtype=torch.int32
    )

    # Call flash attention with window_size
    # window_size is specified as (left, right) - we use symmetric window
    window_size_tuple = (window_size, window_size)

    # Use flash_attn_varlen_func which supports variable length sequences
    # and window_size
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=seq_len,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=seq_len,
        cu_seqlens_k=cu_seqlens,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        window_size=window_size_tuple,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    )

    # Convert back to (batch, num_heads, seq_len, head_dim) format
    output = output.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    return output