import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_backward

logger = logging.getLogger(__name__)


def _scaled_dot_product_cudnn_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
):
    """CUDA cudnn attention backward operator.

    This is a wrapper around the FlagGems scaled_dot_product_attention_backward
    that provides compatibility with PyTorch's _scaled_dot_product_cudnn_attention_backward
    aten operator.

    Args:
        grad_out: Gradient of the attention output
        query: Query tensor (B, H, Q, D)
        key: Key tensor (B, H_k, K, D)
        value: Value tensor (B, H_v, K, D)
        out: Forward pass output
        logsumexp: Logsumexp from forward pass (not used directly in backward)
        philox_seed: Random seed for dropout (not used, dropout_p must be 0)
        philox_offset: Random offset for dropout (not used, dropout_p must be 0)
        attn_bias: Attention bias tensor
        cum_seq_q: Cumulative sequence lengths for query (not supported)
        cum_seq_k: Cumulative sequence lengths for key (not supported)
        max_q: Maximum query sequence length
        max_k: Maximum key sequence length
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Attention scale factor

    Returns:
        Tuple of (grad_query, grad_key, grad_value)
    """
    logger.debug("GEMS SCALED DOT PRODUCT CUDNN ATTENTION BACKWARD")

    # Validate inputs
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"

    # Note: cum_seq_q and cum_seq_k are for variable length sequences
    # which is not supported yet in FlagGems attention backward
    # We assume contiguous sequences (no padding) for now

    # Create M tensor for the backward pass (required by scaled_dot_product_attention_backward)
    # The M tensor stores max values for numerical stability
    # In the backward pass, it will be recomputed via _attn_bwd_preprocess
    M = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2]),
        device=query.device,
        dtype=torch.float32,
    )

    # Call the existing FlagGems attention backward implementation
    dq, dk, dv = scaled_dot_product_attention_backward(
        do=grad_out,
        query=query,
        key=key,
        value=value,
        o=out,
        M=M,
        attn_mask=attn_bias if attn_bias is not None else None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
    )

    return dq, dk, dv