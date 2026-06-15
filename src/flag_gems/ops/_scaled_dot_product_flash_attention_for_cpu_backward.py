import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_backward

logger = logging.getLogger(__name__)


def _scaled_dot_product_flash_attention_for_cpu_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    dropout_p,
    is_causal,
    attn_mask=None,
    scale=None,
):
    """
    Backward pass for scaled_dot_product_flash_attention_for_cpu.

    This operator computes the gradients of query, key, and value tensors
    with respect to the output gradient. It wraps the existing Triton-based
    scaled_dot_product_attention_backward implementation.
    """
    logger.debug("GEMS SCALED DOT PRODUCT FLASH ATTENTION FOR CPU BACKWARD")
    # The existing implementation handles the attn_mask internally
    # and uses Triton kernels for the backward computation
    grad_query, grad_key, grad_value = scaled_dot_product_attention_backward(
        do=grad_out,
        query=query,
        key=key,
        value=value,
        o=out,
        M=logsumexp,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return grad_query, grad_key, grad_value