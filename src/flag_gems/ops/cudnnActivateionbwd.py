import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def cudnnActivateionbwd_kernel(x, dy):
    # Backward of ReLU activation: gradient passes through where x > 0
    return tl.where(x > 0, dy, 0)


def cudnnActivateionbwd(grad_output, self):
    logger.debug("GEMS cudnnActivateionbwd")
    output = cudnnActivateionbwd_kernel(self, grad_output)
    return output