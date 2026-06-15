import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def Grouped_Query_Attention_GQA(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    Grouped Query Attention (GQA) implementation.

    This is a convenience wrapper around scaled_dot_product_attention with GQA enabled.
    GQA is an attention mechanism where the number of key/value heads is smaller than
    the number of query heads. Each group of query heads shares the same key/value heads.

    Args:
        query: Query tensor; shape (batch, num_q_heads, seq_len, head_dim)
        key: Key tensor; shape (batch, num_kv_heads, seq_len, head_dim)
        value: Value tensor; shape (batch, num_kv_heads, seq_len, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (currently must be 0.0)
        is_causal: Whether to apply causal masking
        scale: Optional scale factor

    Returns:
        Attention output tensor; shape (batch, num_q_heads, seq_len, head_dim)
    """
    logger.debug("GEMS Grouped_Query_Attention_GQA")
    return scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,
    )