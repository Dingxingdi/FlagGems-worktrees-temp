import logging

import torch

logger = logging.getLogger(__name__)


def _efficient_attention_forward(
    query,
    key,
    value,
    bias=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    custom_mask_type=0,
    compute_log_sumexp=False,
    *,
    scale=None,
    seqlen_k=None,
    window_size=None,
):
    """Efficient attention forward implementation.

    This is a wrapper around torch.ops.aten._efficient_attention_forward
    that provides the FlagGems interface for this operator.

    Args:
        query: (batch, num_heads, seq_len, head_dim)
        key: (batch, num_heads_k, seq_len_k, head_dim)
        value: (batch, num_heads_k, seq_len_k, head_dim)
        bias: Optional attention bias
        cu_seqlens_q: Cumulative sequence lengths for query
        cu_seqlens_k: Cumulative sequence lengths for key/value
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key/value sequence length
        dropout_p: Dropout probability
        custom_mask_type: Type of custom mask
        compute_log_sumexp: Whether to compute logsumexp
        scale: Optional scale factor
        seqlen_k: Key sequence length
        window_size: Window size for sliding window attention

    Returns:
        tuple: (output, logsumexp, philox_seed, philox_offset, max_seqlen_batch_q, max_seqlen_batch_k)
    """
    logger.debug("GEMS _efficient_attention_forward")

    # Call the native PyTorch implementation
    result = torch.ops.aten._efficient_attention_forward(
        query,
        key,
        value,
        bias,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        custom_mask_type,
        compute_log_sumexp,
        scale=scale,
        seqlen_k=seqlen_k,
        window_size=window_size,
    )

    return result