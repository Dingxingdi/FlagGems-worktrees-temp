import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fused_adam_kernel(
    self_ptr,  # params
    grads_ptr,  # grads
    exp_avgs_ptr,  # exp_avgs
    exp_avg_sqs_ptr,  # exp_avg_sqs
    max_exp_avg_sqs_ptr,  # max_exp_avg_sqs (can be None)
    numel,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad: tl.constexpr,
    maximize: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the element of self/grads/exp_avgs/exp_avg_sqs
    # that this program is responsible for.
    pid = tle.program_id(0)
    # `numel` might not be divisible by `BLOCK_SIZE`. We check that
    # and only compute the portion that is valid.
    start = pid * BLOCK_SIZE
    off = tl.arange(0, BLOCK_SIZE)
    mask = off < numel

    # Load data
    self_ptrs = self_ptr + start + off
    grads_ptrs = grads_ptr + start + off
    exp_avgs_ptrs = exp_avgs_ptr + start + off
    exp_avg_sqs_ptrs = exp_avg_sqs_ptr + start + off

    # Load self (param), grad, exp_avg, exp_avg_sq
    self_val = tl.load(self_ptrs, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grads_ptrs, mask=mask, other=0.0).to(tl.float32)
    exp_avg = tl.load(exp_avgs_ptrs, mask=mask, other=0.0).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sqs_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Apply maximize (negate grad if maximize is True)
    if maximize:
        grad = -grad

    # Adam update logic (fused)
    # Update biased first moment estimate
    exp_avg_new = beta1 * exp_avg + (1 - beta1) * grad
    # Update biased second raw moment estimate
    exp_avg_sq_new = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

    # Compute denominator
    denom = tl.sqrt(exp_avg_sq_new) + eps

    # Apply weight decay (AdamW style - decoupled weight decay)
    if weight_decay > 0.0:
        self_val = self_val - lr * weight_decay * self_val

    # Update parameter
    update = lr * exp_avg_new / denom
    self_new = self_val - update

    # Store updated values
    # Use .to() to convert back to the original dtype
    tl.store(self_ptrs, self_new.to(tl.float32), mask=mask)
    tl.store(exp_avgs_ptrs, exp_avg_new.to(tl.float32), mask=mask)
    tl.store(exp_avg_sqs_ptrs, exp_avg_sq_new.to(tl.float32), mask=mask)

    # Handle max_exp_avg_sqs for AMSGrad
    if amsgrad:
        max_exp_avg_sqs_ptrs = max_exp_avg_sqs_ptr + start + off
        max_exp_avg_sq = tl.load(max_exp_avg_sqs_ptrs, mask=mask, other=0.0).to(tl.float32)
        max_exp_avg_sq_new = tl.maximum(max_exp_avg_sq, exp_avg_sq_new)
        tl.store(
            max_exp_avg_sqs_ptrs,
            max_exp_avg_sq_new.to(tl.float32),
            mask=mask,
        )


def _fused_adam(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
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
    logger.debug("GEMS FUSED_ADAM")

    # Handle the case where inputs are tuples/lists of tensors
    # PyTorch's _fused_adam_ expects tuples
    if isinstance(self, (tuple, list)):
        self = list(self)
    if isinstance(grads, (tuple, list)):
        grads = list(grads)
    if isinstance(exp_avgs, (tuple, list)):
        exp_avgs = list(exp_avgs)
    if isinstance(exp_avg_sqs, (tuple, list)):
        exp_avg_sqs = list(exp_avg_sqs)
    if isinstance(max_exp_avg_sqs, (tuple, list)):
        max_exp_avg_sqs = list(max_exp_avg_sqs)
    if isinstance(state_steps, (tuple, list)):
        state_steps = list(state_steps)

    # Convert lr to float if it's a tensor
    if isinstance(lr, torch.Tensor):
        lr = lr.item()

    # Process each parameter group
    for i in range(len(self)):
        p = self[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        max_exp_avg_sq = max_exp_avg_sqs[i] if amsgrad else None

        # Make sure tensors are contiguous
        p = p.contiguous()
        grad = grad.contiguous()
        exp_avg = exp_avg.contiguous()
        exp_avg_sq = exp_avg_sq.contiguous()
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sq.contiguous()

        numel = p.numel()

        # Define the grid
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

        # Launch the kernel
        _fused_adam_kernel[grid](
            p,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            numel,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
            maximize,
            BLOCK_SIZE=256,
        )

    return None