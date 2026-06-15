import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def cudnnconvfilter_func(x):
    # cudnnconvfilter: identity/passthrough operator
    # Returns the input tensor as-is
    return x


def cudnnconvfilter(A):
    """cudnnconvfilter operator - identity/passthrough."""
    logger.debug("GEMS CUDNNCONVFILTER")
    return cudnnconvfilter_func(A)


def cudnnconvfilter_(A):
    """cudnnconvfilter_ operator - in-place identity/passthrough."""
    logger.debug("GEMS CUDNNCONVFILTER_")
    cudnnconvfilter_func(A, out0=A)
    return A