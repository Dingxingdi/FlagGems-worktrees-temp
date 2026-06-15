import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fused_adam_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    max_exp_avg_sq_ptr,
    n_elements,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    bias_correction1,
    bias_correction2,
    amsgrad: tl.constexpr,
    maximize: tl.constexpr,
    grad_scale_ptr,
    found_inf_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Adam optimizer kernel.

    Computes the Adam optimization step for a single tensor group.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load optional grad_scale and found_inf
    grad_scale = 1.0
    found_inf = 0.0
    if grad_scale_ptr:
        grad_scale = tl.load(grad_scale_ptr).to(tl.float32)
    if found_inf_ptr:
        found_inf = tl.load(found_inf_ptr).to(tl.float32)

    # Skip update if found_inf == 1.0
    skip_update = found_inf == 1.0

    # Load tensors
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if amsgrad:
        max_exp_avg_sq = tl.load(
            max_exp_avg_sq_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)

    # Apply grad scale
    if grad_scale != 1.0:
        grad = grad * grad_scale

    # Apply maximize (negate gradient)
    if maximize:
        grad = -grad

    # Update biased first moment estimate
    # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_new = beta1 * exp_avg + (1.0 - beta1) * grad

    # Update biased second raw moment estimate
    # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
    exp_avg_sq_new = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad

    # Compute bias-corrected estimates
    # denom = sqrt(exp_avg_sq / bias_correction2) + eps
    if amsgrad:
        # max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq)
        max_exp_avg_sq_new = tl.where(exp_avg_sq_new > max_exp_avg_sq, exp_avg_sq_new, max_exp_avg_sq)
        denom = tl.sqrt(max_exp_avg_sq_new / bias_correction2) + eps
    else:
        denom = tl.sqrt(exp_avg_sq_new / bias_correction2) + eps

    # Apply weight decay (AdamW style)
    # grad = grad + weight_decay * param
    if weight_decay != 0.0:
        grad = grad + weight_decay * param

    # Compute step size
    # step_size = lr / bias_correction1
    step_size = lr / bias_correction1

    # Compute parameter update
    # param = param - step_size * exp_avg / denom
    param_new = param - step_size * exp_avg_new / denom

    # Store results (skip if found_inf)
    tl.store(param_ptr + offsets, tl.where(skip_update, param, param_new), mask=mask)
    tl.store(exp_avg_ptr + offsets, tl.where(skip_update, exp_avg, exp_avg_new), mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, tl.where(skip_update, exp_avg_sq, exp_avg_sq_new), mask=mask)

    if amsgrad:
        tl.store(
            max_exp_avg_sq_ptr + offsets,
            tl.where(skip_update, max_exp_avg_sq, max_exp_avg_sq_new),
            mask=mask,
        )


def _compute_bias_corrections(beta1, beta2, state_step):
    """Compute bias corrections for Adam.

    bias_correction1 = 1 - beta1^step
    bias_correction2 = 1 - beta2^step
    """
    step = state_step.item() if isinstance(state_step, torch.Tensor) else state_step
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step
    return bias_correction1, bias_correction2


def _fused_adam(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    """Fused Adam optimizer operator.

    Applies Adam optimization to multiple tensor groups.
    Each tensor group consists of (param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, state_step).

    Args:
        self: List of parameter tensors
        grads: List of gradient tensors
        exp_avgs: List of exponential moving average tensors
        exp_avg_sqs: List of exponential moving average of squared gradients
        max_exp_avg_sqs: List of max exponential moving average of squared gradients (for AMSGrad)
        state_steps: List of state step tensors (scalar)
        lr: Learning rate
        beta1: Exponential moving average coefficient for first moment
        beta2: Exponential moving average coefficient for second moment
        weight_decay: Weight decay coefficient (AdamW style)
        eps: Small constant for numerical stability
        amsgrad: Whether to use AMSGrad variant
        maximize: Whether to maximize (negate gradients)
        grad_scale: Optional gradient scaling factor
        found_inf: Optional tensor indicating if inf/nan was found (skips update if 1.0)

    Returns:
        Tuple of (self_out, grads_out, exp_avgs_out, exp_avg_sqs_out, max_exp_avg_sqs_out)
    """
    logger.debug("GEMS _FUSED_ADAM")

    n_tensors = len(self)
    output_self = []
    output_grads = []
    output_exp_avgs = []
    output_exp_avg_sqs = []
    output_max_exp_avg_sqs = []

    # Process each tensor group
    for i in range(n_tensors):
        param = self[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        max_exp_avg_sq = max_exp_avg_sqs[i] if amsgrad else None
        state_step = state_steps[i]

        # Compute bias corrections in Python
        bias_correction1, bias_correction2 = _compute_bias_corrections(
            beta1, beta2, state_step
        )

        n_elements = param.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        with torch_device_fn.device(param.device):
            _fused_adam_kernel[grid](
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                max_exp_avg_sq if amsgrad else 0,  # Pass nullptr if not amsgrad
                n_elements,
                lr,
                beta1,
                beta2,
                weight_decay,
                eps,
                bias_correction1,
                bias_correction2,
                amsgrad,
                maximize,
                grad_scale,
                found_inf,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        output_self.append(param)
        output_grads.append(grad)
        output_exp_avgs.append(exp_avg)
        output_exp_avg_sqs.append(exp_avg_sq)
        if amsgrad:
            output_max_exp_avg_sqs.append(max_exp_avg_sq)

    return (
        output_self,
        output_grads,
        output_exp_avgs,
        output_exp_avg_sqs,
        output_max_exp_avg_sqs if amsgrad else [],
    )


def _fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    """In-place version of fused Adam optimizer.

    This version modifies tensors in-place and returns nothing.
    """
    logger.debug("GEMS _FUSED_ADAM_")

    n_tensors = len(self)
    BLOCK_SIZE = 1024

    for i in range(n_tensors):
        param = self[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        max_exp_avg_sq = max_exp_avg_sqs[i] if amsgrad else None
        state_step = state_steps[i]

        # Compute bias corrections in Python
        bias_correction1, bias_correction2 = _compute_bias_corrections(
            beta1, beta2, state_step
        )

        n_elements = param.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        with torch_device_fn.device(param.device):
            _fused_adam_kernel[grid](
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                max_exp_avg_sq if amsgrad else 0,
                n_elements,
                lr,
                beta1,
                beta2,
                weight_decay,
                eps,
                bias_correction1,
                bias_correction2,
                amsgrad,
                maximize,
                grad_scale,
                found_inf,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return None