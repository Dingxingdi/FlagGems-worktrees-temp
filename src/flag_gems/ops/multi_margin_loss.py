import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def multi_margin_loss_forward_kernel(
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    out_ptr,
    N,
    C,
    p: tl.constexpr,
    margin: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_C: tl.constexpr = 64,
):
    """
    Multi-margin loss forward kernel.

    For each sample n:
        loss_n = sum_i(max(0, margin - inp[n, target[n]] + inp[n, i])^p) / C
        where i ranges over [0, C-1], i != target[n]
    """
    pid_n = tl.program_id(0)

    # Load target for this sample
    tgt = tl.load(tgt_ptr + pid_n).to(tl.int32)
    assert tgt >= 0 and tgt < C, "Invalid target value"

    # Load weight for target class if provided
    if wgt_ptr is None:
        wgt_tgt = 1.0
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt).to(tl.float32)

    # Load the target score
    inp_tgt_ptr = inp_ptr + pid_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptr).to(tl.float32)

    # Compute loss for this sample
    loss_sum = 0.0

    # Process in blocks over C
    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        mask_c = c_offsets < C

        # Skip the target class
        valid_c = mask_c & (c_offsets != tgt)

        # Load input values
        inp_ptrs = inp_ptr + pid_n * C + c_offsets
        inp_vals = tl.load(inp_ptrs, mask=mask_c, other=0.0).to(tl.float32)

        # Compute max(0, margin - inp_tgt + inp[i])^p
        # For p=1: max(0, margin - inp_tgt + inp[i])
        # For p=2: max(0, margin - inp_tgt + inp[i])^2
        margin_term = margin - inp_tgt + inp_vals
        margin_term = tl.where(margin_term > 0, margin_term, 0.0)

        # Zero out the target position (we use mask_c instead of valid_c for loading,
        # but we need to exclude target from computation)
        margin_term = tl.where(valid_c, margin_term, 0.0)

        if p == 1:
            loss_term = margin_term
        else:
            # p == 2
            loss_term = margin_term * margin_term

        # Apply weight
        loss_term = loss_term * wgt_tgt

        # Sum valid terms (invalid terms are already 0 due to mask)
        loss_sum = loss_sum + tl.sum(loss_term)

    # Normalize by C (number of classes)
    # For 'none': each sample loss is loss_sum / C
    # For 'mean': final result is sum(loss_sum) / C / N
    # For 'sum': final result is sum(loss_sum) / C
    loss_sum = loss_sum / C

    # Store result
    if reduction == 0:
        # none: store per-sample loss
        tl.store(out_ptr + pid_n, loss_sum)
    else:
        # mean/sum: use atomic add to accumulate across blocks
        tl.atomic_add(out_ptr, loss_sum, sem="relaxed")


@libentry()
@triton.jit
def multi_margin_loss_backward_kernel(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    wgt_ptr,
    inp_grad_ptr,
    N,
    C,
    p: tl.constexpr,
    margin: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_C: tl.constexpr = 64,
):
    """
    Multi-margin loss backward kernel.

    Gradient with respect to input:
    For sample n and class i:
        if i == target[n]:
            grad[n, i] = -sum_j(w[target[n]] * p * (margin - inp[n, target[n]] + inp[n, j])^(p-1)) / C
        else:
            grad[n, i] = w[target[n]] * p * (margin - inp[n, target[n]] + inp[n, i])^(p-1) / C

    For mean reduction, there's an additional division by N:
        grad = grad / N

    where j ranges over all classes != target[n], and only terms where (margin - inp[n, target[n]] + inp[n, j]) > 0 contribute.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Load target for this sample
    tgt = tl.load(tgt_ptr + pid_n).to(tl.int32)

    # Load weight for target class if provided
    if wgt_ptr is None:
        wgt_tgt = 1.0
    else:
        wgt_tgt = tl.load(wgt_ptr + tgt).to(tl.float32)

    # Load the target score
    inp_tgt_ptr = inp_ptr + pid_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptr).to(tl.float32)

    # Compute gradient normalization factor
    # Loss is normalized by C, so gradients are also normalized by C
    # For mean: additional division by N
    if reduction == 1:  # mean
        scale = wgt_tgt * p / C / N
    else:  # sum or none
        scale = wgt_tgt * p / C

    # Process this block of classes
    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    # Check which indices are valid (not target class)
    is_target = c_offsets == tgt
    valid_c = mask_c & ~is_target

    # Load input values for these classes
    inp_ptrs = inp_ptr + pid_n * C + c_offsets
    inp_vals = tl.load(inp_ptrs, mask=mask_c, other=0.0).to(tl.float32)

    # Compute margin - inp_tgt + inp[i]
    margin_term = margin - inp_tgt + inp_vals
    # Only positive terms contribute
    margin_term = tl.where(margin_term > 0, margin_term, 0.0)
    # Zero out the target position
    margin_term = tl.where(valid_c, margin_term, 0.0)

    if p == 1:
        # For p=1: derivative is 1 if margin_term > 0, else 0
        grad_term = tl.where(margin_term > 0, 1.0, 0.0)
    else:
        # For p=2: derivative is 2 * margin_term if margin_term > 0, else 0
        grad_term = tl.where(margin_term > 0, 2.0 * margin_term, 0.0)

    # For non-target classes: grad = scale * grad_term
    grad_non_target = grad_term * scale

    # Store gradients for non-target classes
    inp_grad_ptrs = inp_grad_ptr + pid_n * C + c_offsets
    tl.store(inp_grad_ptrs, grad_non_target, mask=valid_c)

    # Handle gradient for target class: sum of negative gradients of all other classes
    if pid_c == 0:
        # Compute sum of positive contributions from all classes != target
        # grad_term is already 0 for target position due to valid_c mask
        sum_positive = tl.sum(grad_term)
        grad_target = -sum_positive * scale

        # Store target gradient
        tgt_grad_ptr = inp_grad_ptr + pid_n * C + tgt
        tl.store(tgt_grad_ptr, grad_target)


def multi_margin_loss_forward(
    self, target, p=1, margin=1.0, weight=None, reduction=1
):
    """
    Multi-margin loss forward function.

    Args:
        self: Input tensor of shape (N, C) or (C)
        target: Target tensor of shape (N,) or ()
        p: Power term, only 1 or 2 supported
        margin: Margin value
        weight: Optional weight tensor of shape (C,)
        reduction: 0=none, 1=mean, 2=sum
    """
    logger.debug("GEMS multi_margin_loss FWD")

    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    assert target.numel() == N, "Invalid target size"
    assert p in (1, 2), "Only p=1 and p=2 are supported"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    # reduction: 0-None, 1-mean, 2-sum
    if reduction == 0:
        out = torch.empty([N], dtype=torch.float32, device=self.device)
    else:
        out = torch.zeros([], dtype=torch.float32, device=self.device)

    grid = lambda meta: (N,)
    with torch_device_fn.device(self.device):
        multi_margin_loss_forward_kernel[grid](
            self,
            target,
            weight,
            out,
            N,
            C,
            p,
            margin,
            reduction,
        )

    # For mean reduction, divide by N (number of samples)
    if reduction == 1:
        out = out / N

    out = out.to(self.dtype)
    return out


def multi_margin_loss_backward(
    grad_output, self, target, p=1, margin=1.0, weight=None, reduction=1
):
    """
    Multi-margin loss backward function.
    """
    logger.debug("GEMS multi_margin_loss BWD")

    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    grid = lambda meta: (N, triton.cdiv(C, meta["BLOCK_C"]))
    with torch_device_fn.device(self.device):
        multi_margin_loss_backward_kernel[grid](
            grad_output,
            self,
            target,
            weight,
            grad_input,
            N,
            C,
            p,
            margin,
            reduction,
        )

    return grad_input