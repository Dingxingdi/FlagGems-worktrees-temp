import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _prelu_kernel_func(x, weight):
    return tl.where(x >= 0, x, weight * x)


def _prelu_kernel(A, B):
    logger.debug("GEMS _prelu_kernel")
    return _prelu_kernel_func(A, B)


def _prelu_kernel_(A, B):
    logger.debug("GEMS _prelu_kernel_")
    _prelu_kernel_func(A, B, out0=A)
    return A