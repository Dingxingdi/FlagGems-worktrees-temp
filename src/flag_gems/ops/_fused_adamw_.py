import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fused_adamw_kernel(
    params_ptr,
    grads_ptr,
    exp_avgs_ptr,
    exp_avg_sqs_ptr,
    max_exp_avg_sqs_ptr,
    numel,
    lr,
    one_minus_beta1,
    one_minus_beta2,
    weight_decay,
    eps,
    bias_correction1,
    bias_correction2,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the element of the tensors it should compute.
    pid = tle.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < numel

    # Load data
    param = tl.load(params_ptr + off, mask=mask, other=0.0).to(tl.float32)
    exp_avg = tl.load(exp_avgs_ptr + off, mask=mask, other=0.0).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sqs_ptr + off, mask=mask, other=0.0).to(tl.float32)

    # Compute the update using precomputed bias correction factors
    step_size = lr / bias_correction1
    denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps

    # Compute update with weight decay (AdamW style - decoupled weight decay)
    update = (exp_avg / denom) + (weight_decay * param)
    param = param - step_size * update

    # Store updated value
    tl.store(params_ptr + off, param, mask=mask)

    # Compute the update using precomputed bias correction factors
    # Step size: lr / bias_correction1
    # Adam update: (m / bias_correction1) / (sqrt(v / bias_correction2) + eps)
    # With weight decay: + weight_decay * param
    step_size = lr / bias_correction1
    denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps

    # Compute update with weight decay (AdamW style - decoupled weight decay)
    update = (exp_avg / denom) + (weight_decay * param)
    param = param - step_size * update

    # If amsgrad, update max_exp_avg_sqs
    if amsgrad:
        max_exp_avg_sq = tl.load(max_exp_avg_sqs_ptr + off, mask=mask, other=0.0).to(
            tl.float32
        )
        max_exp_avg_sq = tl.where(exp_avg_sq > max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq)
        # Recompute denom with max for amsgrad
        denom_amsgrad = tl.sqrt(max_exp_avg_sq / bias_correction2) + eps
        update_amsgrad = (exp_avg / denom_amsgrad) + (weight_decay * param)
        param = param - step_size * update_amsgrad
        # Store updated max_exp_avg_sqs
        tl.store(max_exp_avg_sqs_ptr + off, max_exp_avg_sq, mask=mask)

    # Store updated values
    tl.store(params_ptr + off, param, mask=mask)
    tl.store(grads_ptr + off, grad, mask=mask)
    tl.store(exp_avgs_ptr + off, exp_avg, mask=mask)
    tl.store(exp_avg_sqs_ptr + off, exp_avg_sq, mask=mask)


def _fused_adamw_(
    params,
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
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("GEMS FUSED_ADAMW")

    # Handle the case where inputs are tuples/lists of tensors
    if isinstance(params, (tuple, list)):
        params = params[0] if len(params) > 0 else params
    if isinstance(grads, (tuple, list)):
        grads = grads[0] if len(grads) > 0 else grads
    if isinstance(exp_avgs, (tuple, list)):
        exp_avgs = exp_avgs[0] if len(exp_avgs) > 0 else exp_avgs
    if isinstance(exp_avg_sqs, (tuple, list)):
        exp_avg_sqs = exp_avg_sqs[0] if len(exp_avg_sqs) > 0 else exp_avg_sqs
    if isinstance(max_exp_avg_sqs, (tuple, list)):
        max_exp_avg_sqs = max_exp_avg_sqs[0] if len(max_exp_avg_sqs) > 0 else max_exp_avg_sqs
    if isinstance(state_steps, (tuple, list)):
        state_steps = state_steps[0] if len(state_steps) > 0 else state_steps

    # Handle None max_exp_avg_sqs for non-amsgrad
    if max_exp_avg_sqs is None:
        max_exp_avg_sqs = torch.zeros_like(exp_avg_sqs)

    # Apply gradient negation if maximize is True
    if maximize:
        grads = -grads

    # Compute bias correction factors from state_steps
    if state_steps is not None and state_steps.numel() > 0:
        max_step = state_steps.max().item()
    else:
        max_step = 0

    bias_correction1 = 1.0 - beta1 ** (max_step + 1)
    bias_correction2 = 1.0 - beta2 ** (max_step + 1)

    if bias_correction1 == 0:
        bias_correction1 = 1.0
    if bias_correction2 == 0:
        bias_correction2 = 1.0

    # Compute one_minus_beta values
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2

    # Update exp_avgs and exp_avg_sqs
    exp_avgs.add_(grads, alpha=one_minus_beta1)
    exp_avg_sqs.add_(grads * grads, alpha=one_minus_beta2)

    # For amsgrad, update max_exp_avg_sqs
    if amsgrad:
        torch.maximum(max_exp_avg_sqs, exp_avg_sqs, out=max_exp_avg_sqs)
        denom = torch.sqrt(max_exp_avg_sqs / bias_correction2) + eps
    else:
        denom = torch.sqrt(exp_avg_sqs / bias_correction2) + eps

    # Compute parameter update and apply
    step_size = lr / bias_correction1
    update = (exp_avgs / denom) + (weight_decay * params)
    params.sub_(update, alpha=step_size)

    return None


from flag_gems.runtime import torch_device_fn