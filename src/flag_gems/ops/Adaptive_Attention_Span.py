import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def adaptive_attention_span_forward(x):
    """Compute adaptive attention span using sigmoid activation.

    This operation maps input values to (0, 1) range, representing
    a normalized attention span. This is commonly used in Transformer
    models with adaptive attention mechanisms.
    """
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + exp2(-x.to(tl.float32) * log2e))


def adaptive_attention_span(A):
    """Compute adaptive attention span for input tensor.

    Args:
        A: Input tensor of any shape

    Returns:
        Tensor of the same shape with values in (0, 1) range
    """
    logger.debug("GEMS ADAPTIVE_ATTENTION_SPAN")
    return adaptive_attention_span_forward(A)


def adaptive_attention_span_(A):
    """In-place version of adaptive_attention_span."""
    logger.debug("GEMS ADAPTIVE_ATTENTION_SPAN_")
    adaptive_attention_span_forward(A, out0=A)
    return A