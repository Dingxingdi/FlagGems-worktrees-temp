import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def bce_backward_kernel_no_weight(grad_output, self, target, reduction):
    # BCE backward formula when self is probability (not logits):
    # grad = grad_output * (self - target) / (self * (1 - self))
    # We need to handle the case where self is close to 0 or 1
    p = self.to(tl.float32)
    y = target.to(tl.float32)
    go = grad_output.to(tl.float32)

    denom = p * (1.0 - p)
    # Avoid division by zero: when p is very close to 0 or 1, use a safe formula
    # For p near 0: (p - y) / p ≈ -y when p ≈ 0
    # For p near 1: (p - y) / (1-p) ≈ y when p ≈ 1
    # We use a simple approach: clamp denom to avoid zero
    denom = tl.where(denom < 1e-6, 1e-6, denom)
    grad = go * (p - y) / denom
    return grad


@pointwise_dynamic(is_tensor=[True, True, True, True, False], promotion_methods=[(0, 1, 2, 3, "DEFAULT")])
@triton.jit
def bce_backward_kernel_weight(grad_output, self, target, weight, reduction):
    # BCE backward formula with weight when self is probability:
    # grad = grad_output * weight * (self - target) / (self * (1 - self))
    p = self.to(tl.float32)
    y = target.to(tl.float32)
    w = weight.to(tl.float32)
    go = grad_output.to(tl.float32)

    denom = p * (1.0 - p)
    denom = tl.where(denom < 1e-6, 1e-6, denom)
    grad = go * w * (p - y) / denom
    return grad


def binary_cross_entropy_backward(grad_output, self, target, weight=None, reduction=1):
    logger.debug("GEMS BINARY_CROSS_ENTROPY_BACKWARD")
    n = self.numel()

    if weight is not None:
        grad = bce_backward_kernel_weight(grad_output, self, target, weight, reduction)
    else:
        grad = bce_backward_kernel_no_weight(grad_output, self, target, reduction)

    # Handle reduction
    # reduction: 0='none', 1='mean', 2='sum'
    if reduction == 1:  # mean
        grad = grad / n
    # reduction == 0 (none) or reduction == 2 (sum): no change needed

    return grad