# FlagGems scaled_dot_product_attention operator implementation
#
# This file provides a wrapper for the scaled_dot_product_attention operator
# which is implemented in attention.py

import logging

from flag_gems.ops.attention import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)

logger = logging.getLogger(__name__)

__all__ = [
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
]