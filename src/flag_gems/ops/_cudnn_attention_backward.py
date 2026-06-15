import logging

import torch

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
    scale=None,
):
    """Backward pass for scaled dot product attention using cuDNN backend.

    This operator computes the gradients of query, key, and value tensors
    with respect to the loss gradient.
    """
    logger.debug("GEMS SCALED DOT PRODUCT CUDNN ATTENTION BACKWARD")

    # Since FlagGems doesn't intercept this operator (removed from _FULL_CONFIG),
    # this function acts as a wrapper that ensures the cuDNN implementation is used.
    # When called within flag_gems.use_gems() context, the aten operator dispatch
    # will use PyTorch's default cuDNN implementation.
    return torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
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
        dropout_p,
        is_causal,
        scale=scale,
    )


# Register as the aten operator
_aten_op = torch._ops.ops.aten._scaled_dot_product_cudnn_attention_backward.default