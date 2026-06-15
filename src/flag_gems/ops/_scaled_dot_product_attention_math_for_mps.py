import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.ops.attention import scaled_dot_product_attention as gems_sdpa

logger = logging.getLogger(__name__)


def _scaled_dot_product_attention_math_for_mps(query, key, value):
    """
    Scaled Dot Product Attention Math implementation for MPS backend.

    This is a wrapper that uses the standard GEMS scaled_dot_product_attention
    implementation with default parameters (no mask, no dropout, no causal).
    This matches the behavior of PyTorch's _scaled_dot_product_attention_math_for_mps
    which uses the math backend for attention computation.
    """
    logger.debug("GEMS SCALED_DOT_PRODUCT_ATTENTION_MATH_FOR_MPS")

    # Call GEMS scaled_dot_product_attention with default parameters
    # This uses the math-based implementation (not flash attention)
    output = gems_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,  # Will use default 1/sqrt(head_dim)
        enable_gqa=False,
    )

    return output