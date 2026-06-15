import logging

import torch

from flag_gems.ops.attention import flash_attention_forward

logger = logging.getLogger(__name__)


def FlashAttention_V2(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    """FlashAttention V2 operator.

    This is a simplified wrapper around FlashAttention that provides a simpler
    interface similar to PyTorch's scaled_dot_product_flash_attention.

    Args:
        query: Query tensor of shape (batch, num_heads, seqlen_q, head_dim)
        key: Key tensor of shape (batch, num_heads_k, seqlen_k, head_dim)
        value: Value tensor of shape (batch, num_heads_k, seqlen_k, head_dim)
        dropout_p: Dropout probability (default: 0.0)
        is_causal: Whether to apply causal masking (default: False)
        return_debug_mask: Whether to return attention weights (default: False)
        scale: Optional scale factor for attention scores (default: 1/sqrt(head_dim))

    Returns:
        output: Attention output of shape (batch, num_heads, seqlen_q, head_dim)
    """
    logger.debug("GEMS FlashAttention_V2")

    # Validate input shapes
    assert query.dim() == 4, f"query must be 4D, got {query.dim()}D"
    assert key.dim() == 4, f"key must be 4D, got {key.dim()}D"
    assert value.dim() == 4, f"value must be 4D, got {value.dim()}D"

    batch_size, num_heads, seqlen_q, head_dim = query.shape
    _, num_heads_k, seqlen_k, _ = key.shape

    # Transpose to (batch, seqlen, num_heads, head_dim) format for flash_attention_forward
    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)
    value_t = value.transpose(1, 2)

    # Call the underlying flash attention implementation
    # flash_attention_forward returns (output, lse, philox_seed, philox_offset, debug_attn_mask)
    result = flash_attention_forward(
        query_t,
        key_t,
        value_t,
        cumulative_sequence_length_q=None,
        cumulative_sequence_length_k=None,
        max_q=seqlen_q,
        max_k=seqlen_k,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )

    # Transpose back to (batch, num_heads, seqlen_q, head_dim) format
    output = result[0].transpose(1, 2)

    # Return only the output tensor
    return output