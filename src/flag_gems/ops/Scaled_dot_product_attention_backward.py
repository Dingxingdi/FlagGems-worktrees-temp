import logging

from flag_gems.ops.attention import scaled_dot_product_attention_backward

logger = logging.getLogger(__name__)


def Scaled_dot_product_attention_backward(
    grad_output,
    query,
    key,
    value,
    output,
    logsumexp,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Backward function for Scaled Dot Product Attention.

    This function computes the gradients of query, key, and value tensors
    with respect to the output of scaled_dot_product_attention.

    Args:
        grad_output: Gradient of the attention output (do)
        query: Query tensor
        key: Key tensor
        value: Value tensor
        output: Original attention output (o)
        logsumexp: Logsumexp from forward pass (M)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Optional scale factor
        enable_gqa: Whether to enable grouped query attention

    Returns:
        Tuple of (grad_query, grad_key, grad_value)
    """
    logger.debug("GEMS SCALED_DOT_PRODUCT_ATTENTION_BACKWARD")
    return scaled_dot_product_attention_backward(
        grad_output,
        query,
        key,
        value,
        output,
        logsumexp,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
    )