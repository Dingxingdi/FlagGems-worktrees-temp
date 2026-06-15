import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def huber_loss_backward_kernel(grad_output, self, target, delta):
    # Compute diff = self - target
    diff = (self - target).to(tl.float32)
    grad_out = grad_output.to(tl.float32)
    d = delta.to(tl.float32)
    abs_diff = tl.abs(diff)
    # Huber loss gradient:
    # - If |diff| <= delta: gradient = grad_output * diff
    # - If |diff| > delta: gradient = grad_output * delta * sign(diff)
    sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))
    gradient = tl.where(abs_diff <= d, grad_out * diff, grad_out * d * sign)
    return gradient


def huber_loss_backward(grad_output, self, target, reduction, delta):
    logger.debug("GEMS HUBER_LOSS_BACKWARD")
    # reduction: 0 = none, 1 = mean, 2 = sum
    # For reduction=1 (mean), we need to divide by n
    result = huber_loss_backward_kernel(grad_output, self, target, delta)
    if reduction == 1:  # mean
        n = self.numel()
        result = result / n
    return result