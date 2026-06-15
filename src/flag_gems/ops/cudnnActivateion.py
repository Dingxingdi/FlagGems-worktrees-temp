import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def cudnnActivateion_forward(x):
    # Simple ReLU-like activation: max(0, x)
    return tl.where(x > 0, x, 0)


def cudnnActivateion(self):
    logger.debug("GEMS cudnnActivateion FORWARD")
    output = cudnnActivateion_forward(self)
    return output


def cudnnActivateion_(A):
    logger.debug("GEMS cudnnActivateion_ FORWARD")
    out = cudnnActivateion_forward(A, out0=A)
    return out