import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def detach_kernel(x):
    return x


def detach_(A):
    """In-place detach: returns a tensor that doesn't require gradients."""
    logger.debug("GEMS DETACH_")
    detach_kernel(A, out0=A)
    return A


def detach(A):
    """Functional detach: returns a tensor that doesn't require gradients."""
    logger.debug("GEMS DETACH")
    return detach_kernel(A)