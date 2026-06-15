import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def fused_cross_entropy_forward_kernel_1(
    output_ptr,
    input_ptr,
    target_ptr,
    weight_ptr,
    ignore_index,
    M,
    N,
    BLOCK_N: tl.constexpr,
    reduction: tl.constexpr,
):
    """
    Fused Cross Entropy Forward Kernel - handles BLOCK_M=1 case
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx

    # Load target
    target = tl.load(target_ptr + row_offset)
    ignore_mask = target != ignore_index

    # Load input row - only load valid columns
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    input_ptrs = input_ptr + row_offset * N + col_offsets
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # Compute log_softmax with proper masking
    # First get max only from valid entries
    m = tl.max(inp)  # max - this should work since -inf is smaller than any real value

    # Shift and compute exp - use mask to zero out invalid entries
    inp_shifted = tl.where(mask, inp - m, -float("inf"))
    exp_shifted = tl.exp(inp_shifted)
    # Apply mask again to zero out invalid entries
    exp_shifted_masked = tl.where(mask, exp_shifted, 0.0)
    exp_sum = tl.sum(exp_shifted_masked)

    # Compute log_softmax
    log_softmax = tl.where(mask, inp_shifted - tl.log(exp_sum), 0.0)

    # Get the logit for target class
    target_offset = row_offset * N + target
    target_logit = tl.load(input_ptr + target_offset)

    # Compute log_softmax for target: logits[target] - max - log(sum(exp))
    target_shifted = target_logit - m
    log_softmax_target = target_shifted - tl.log(exp_sum)

    loss = -log_softmax_target

    # Apply weight if provided
    if weight_ptr is not None:
        wgt = tl.load(weight_ptr + target).to(tl.float32)
        loss = loss * wgt
    else:
        wgt = 1.0

    # Store based on reduction mode
    if reduction == 0:  # none
        tl.store(output_ptr + row_offset, loss)
    elif reduction == 1:  # mean
        tl.atomic_add(output_ptr, loss, sem="relaxed")
        tl.atomic_add(output_ptr + 1, wgt, sem="relaxed")
    else:  # sum
        tl.atomic_add(output_ptr, loss, sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def fused_cross_entropy_backward_kernel_1(
    grad_input_ptr,
    input_ptr,
    target_ptr,
    weight_ptr,
    ignore_index,
    grad_output_ptr,
    total_weight_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
    reduction: tl.constexpr,
):
    """
    Fused Cross Entropy Backward Kernel - handles BLOCK_M=1 case
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx

    # Load target
    target = tl.load(target_ptr + row_offset)
    ignore_mask = target != ignore_index

    # Load input row
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    input_ptrs = input_ptr + row_offset * N + col_offsets
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # Compute softmax with proper masking
    m = tl.max(inp)
    inp_shifted = tl.where(mask, inp - m, -float("inf"))
    exp_shifted = tl.exp(inp_shifted)
    exp_shifted_masked = tl.where(mask, exp_shifted, 0.0)
    exp_sum = tl.sum(exp_shifted_masked)
    softmax = tl.where(mask, exp_shifted / exp_sum, 0.0)

    # Get weight
    if weight_ptr is not None:
        wgt_target = tl.load(weight_ptr + target).to(tl.float32)
    else:
        wgt_target = 1.0

    # Get grad_output
    if reduction == 0:
        grad_out = tl.load(grad_output_ptr + row_offset).to(tl.float32)
    else:
        grad_out = tl.load(grad_output_ptr).to(tl.float32)

    if reduction == 1:
        total_w = tl.load(total_weight_ptr).to(tl.float32)
    else:
        total_w = 1.0

    scale = grad_out * wgt_target / total_w

    # Compute gradient: softmax - one_hot
    for c in range(BLOCK_N):
        if col_offsets[c] < N:
            is_target = (target == col_offsets[c]) & ignore_mask
            grad = tl.where(is_target, softmax[c] - 1.0, softmax[c])
            grad = grad * scale
            tl.store(grad_input_ptr + row_offset * N + col_offsets[c], grad)


# Helper function to select BLOCK_N based on N
def get_block_n(N):
    if N <= 64:
        return 64
    elif N <= 128:
        return 128
    elif N <= 256:
        return 256
    elif N <= 512:
        return 512
    elif N <= 1024:
        return 1024
    elif N <= 2048:
        return 2048
    elif N <= 4096:
        return 4096
    elif N <= 8192:
        return 8192
    elif N <= 16384:
        return 16384
    elif N <= 32768:
        return 32768
    else:
        return 65536


def fused_cross_entropy_forward(self, target, weight=None, reduction="mean", ignore_index=-100):
    """
    Fused Cross Entropy Forward
    """
    logger.debug("GEMS Fused Cross Entropy FWD")

    # Convert reduction string to int
    if reduction == "none":
        reduction_int = 0
    elif reduction == "mean":
        reduction_int = 1
    elif reduction == "sum":
        reduction_int = 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    assert self.ndim == 2, "Invalid input ndim for fused cross entropy, expected 2D input (N, C)"
    M, N = self.shape
    assert target.numel() == M, f"Invalid target size: {target.numel()} vs {M}"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    # Allocate output based on reduction
    if reduction_int == 0:
        out = torch.empty(target.shape, dtype=self.dtype, device=self.device)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    elif reduction_int == 1:
        out = torch.zeros([2], dtype=torch.float32, device=self.device)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)
    else:
        out = torch.zeros([], dtype=torch.float32, device=self.device)
        total_weight = torch.empty([], dtype=self.dtype, device=self.device)

    # Select block size - needs to be >= max number of classes
    BLOCK_N = get_block_n(N)

    grid = (M, )
    with torch_device_fn.device(self.device):
        fused_cross_entropy_forward_kernel_1[grid](
            out,
            self,
            target,
            weight,
            ignore_index,
            M,
            N,
            BLOCK_N,
            reduction_int,
        )

    # Process output based on reduction
    if reduction_int == 0:
        output = out
    elif reduction_int == 1:
        torch.cuda.synchronize()
        total_loss = out[0]
        total_weight_val = out[1]
        if total_weight_val.item() > 0:
            output = (total_loss / total_weight_val).to(self.dtype)
        else:
            output = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        total_weight = total_weight_val.to(self.dtype)
    else:
        output = out.to(self.dtype)

    return output, total_weight


def fused_cross_entropy_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction="mean",
    ignore_index=-100,
    total_weight=None,
):
    """
    Fused Cross Entropy Backward
    """
    logger.debug("GEMS Fused Cross Entropy BWD")

    # Convert reduction string to int
    if reduction == "none":
        reduction_int = 0
    elif reduction == "mean":
        reduction_int = 1
    elif reduction == "sum":
        reduction_int = 2
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    M, N = self.shape

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    BLOCK_N = get_block_n(N)

    grid = (M, )
    with torch_device_fn.device(self.device):
        fused_cross_entropy_backward_kernel_1[grid](
            grad_input,
            self,
            target,
            weight,
            ignore_index,
            grad_output,
            total_weight,
            M,
            N,
            BLOCK_N,
            reduction_int,
        )

    return grad_input


# Create the user-facing function that matches torch.Fused_Cross_Entropy
def Fused_Cross_Entropy(logits, target, weight=None, reduction="mean", ignore_index=-100):
    """
    Fused Cross Entropy Loss
    """
    logger.debug("GEMS Fused_Cross_Entropy")
    return fused_cross_entropy_forward(logits, target, weight, reduction, ignore_index)