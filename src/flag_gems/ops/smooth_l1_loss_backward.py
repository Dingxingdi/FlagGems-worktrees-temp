import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def smooth_l1_loss_backward_kernel(grad_output, self, target, reduction, beta):
    diff = self - target
    abs_diff = tl.abs(diff)
    # smooth_l1 gradient: if |diff| < beta: diff/beta, else: sign(diff)
    # sign(diff) = 1.0 if diff > 0, 0.0 if diff == 0, -1.0 if diff < 0
    sign_diff = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))
    elementwise_grad = tl.where(abs_diff < beta, diff / beta, sign_diff)
    # Apply grad_output scaling
    result = elementwise_grad * grad_output
    return result


def smooth_l1_loss_backward(grad_output, self, target, reduction, beta):
    logger.debug("GEMS SMOOTH_L1_LOSS_BACKWARD")
    if beta == 0:
        # Avoid division by zero
        return torch.zeros_like(self)

    # Handle device mismatch
    if target.device != self.device:
        target = target.to(self.device)
    if grad_output.device != self.device:
        grad_output = grad_output.to(self.device)

    result = smooth_l1_loss_backward_kernel(grad_output, self, target, reduction, beta)

    # Apply mean reduction if needed (reduction=1 means mean)
    if reduction == 1:
        n = result.numel()
        result = result / n

    return result