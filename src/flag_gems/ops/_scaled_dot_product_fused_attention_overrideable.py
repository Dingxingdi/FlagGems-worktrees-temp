import logging

import torch
from flag_gems.ops.attention import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def _scaled_dot_product_fused_attention_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Overrideable scaled dot product attention.

    This is a wrapper around scaled_dot_product_attention that can be
    used as an alternative aten operator for attention computation.
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_FUSED_ATTENTION_OVERRIDEABLE")
    return scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )