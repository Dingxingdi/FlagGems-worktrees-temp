import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_backward

logger = logging.getLogger(__name__)


def _scaled_dot_product_flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    scale=None,
):
    """Backward pass for scaled_dot_product_flash_attention.

    This is a wrapper that adapts the flash attention backward API to the
    existing scaled_dot_product_attention_backward implementation.

    Args:
        grad_out: Gradient of the output tensor
        query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, num_kv_heads, seq_len_k, head_dim)
        value: Value tensor of shape (batch, num_kv_heads, seq_len_k, head_dim)
        out: Output tensor from forward pass
        logsumexp: Logsumexp from forward pass
        cum_seq_q: Cumulative sequence lengths for query
        cum_seq_k: Cumulative sequence lengths for key/value
        max_q: Maximum query sequence length
        max_k: Maximum key/value sequence length
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        philox_seed: Random seed for dropout
        philox_offset: Random offset for dropout
        scale: Optional scale factor

    Returns:
        Tuple of (grad_query, grad_key, grad_value)
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_FLASH_ATTENTION_BACKWARD")

    # Delegate to the existing scaled_dot_product_attention_backward
    # The flash attention backward API is similar, we just pass the relevant arguments
    dq, dk, dv = scaled_dot_product_attention_backward(
        do=grad_out,
        query=query,
        key=key,
        value=value,
        o=out,
        M=logsumexp,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    return dq, dk, dv