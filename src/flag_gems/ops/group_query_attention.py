import logging

import torch
import torch.nn.functional as F

from flag_gems.ops.attention import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def group_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    scale: float = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Grouped Query Attention (GQA) operation.

    This is a wrapper around scaled_dot_product_attention with GQA enabled.
    GQA is an attention mechanism where different query heads share fewer
    key/value heads, reducing memory and compute costs.

    Args:
        query: Query tensor of shape (batch, q_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, kv_heads, seq_len, head_dim)
        value: Value tensor of shape (batch, kv_heads, seq_len, head_dim)
        attn_mask: Optional attention mask
        scale: Optional scale factor for attention scores
        is_causal: Whether to use causal masking
        dropout_p: Dropout probability (currently must be 0.0)

    Returns:
        Output tensor of shape (batch, q_heads, seq_len, head_dim)
    """
    logger.debug("GEMS GROUP_QUERY_ATTENTION")
    # Validate inputs
    assert query.dim() == 4, f"query must be 4D, got {query.dim()}D"
    assert key.dim() == 4, f"key must be 4D, got {key.dim()}D"
    assert value.dim() == 4, f"value must be 4D, got {value.dim()}D"

    # Ensure q_heads is divisible by kv_heads for GQA
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    assert q_heads % kv_heads == 0, (
        f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads})"
    )

    # Use scaled_dot_product_attention with GQA enabled
    output = scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,
    )

    return output