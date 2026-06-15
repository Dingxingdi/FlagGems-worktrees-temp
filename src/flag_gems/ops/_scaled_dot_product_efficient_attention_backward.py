import logging

import torch

logger = logging.getLogger(__name__)


def scaled_dot_product_efficient_attention_backward(
    grad_out_,
    query,
    key,
    value,
    attn_bias,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    dropout_p,
    grad_input_mask,
    is_causal=False,
    scale=None,
):
    logger.debug("GEMS SCALED DOT PRODUCT EFFICIENT ATTENTION BACKWARD")
    # Delegate to PyTorch's implementation
    result = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out_,
        query,
        key,
        value,
        attn_bias,
        out,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask,
        is_causal,
        scale=scale,
    )
    return result