import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def dropout_add_layernorm_kernel(
    in_ptr1,
    in_ptr2,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,
    out_rstd_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    """Fused kernel: input + residual -> layer_norm"""
    pid = tle.program_id(0)

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    # Load input and residual
    x1 = tl.load(in_ptr1 + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    x2 = tl.load(in_ptr2 + pid * N + n_offsets, mask, other=0.0).to(tl.float32)

    # Add: residual = input + residual
    x = x1 + x2

    # LayerNorm computation
    m = tl.sum(x) / N
    d = x - m
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    if weight_ptr is None:
        w = 1
    else:
        w = tl.load(weight_ptr + n_offsets, mask=mask)
    if bias_ptr is None:
        b = 0
    else:
        b = tl.load(bias_ptr + n_offsets, mask=mask)

    out = (x - m) * rstd * w + b

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_backward"),
    key=["M", "N"],
)
@triton.jit
def dropout_add_layernorm_backward_kernel(
    dY,
    X1,
    X2,
    Mean,
    Rstd,
    W,
    dX1,
    dX2,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """Backward kernel for Dropout_Add_LayerNorm"""
    pid = tle.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M
    dY += pid * N
    X1 += pid * N
    X2 += pid * N
    dX1 += pid * N
    dX2 += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean, mask=row_mask).to(tl.float32)
    rstd = tl.load(Rstd, mask=row_mask).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask

        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)

        x1 = tl.load(X1 + cols[None, :], mask).to(tl.float32)
        x2 = tl.load(X2 + cols[None, :], mask).to(tl.float32)
        x = x1 + x2
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd

        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)

        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask

        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)

        x1 = tl.load(X1 + cols[None, :], mask).to(tl.float32)
        x2 = tl.load(X2 + cols[None, :], mask).to(tl.float32)
        x = x1 + x2
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd

        if W is None:
            w = 1
        else:
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)

        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)

        # Split gradient equally between input and residual
        dx1 = dx * 0.5
        dx2 = dx * 0.5

        tl.store(dX1 + cols, dx1, mask=mask)
        tl.store(dX2 + cols, dx2, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_bias_backward"),
    key=["N"],
)
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X1,
    X2,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    """Backward kernel for weight and bias"""
    pid = tle.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    dY += pid[None, :]
    X1 += pid[None, :]
    X2 += pid[None, :]
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask and col_mask[None, :]

        dy = tl.load(dY + rows * N, mask).to(tl.float32)

        x1 = tl.load(X1 + rows * N, mask).to(tl.float32)
        x2 = tl.load(X2 + rows * N, mask).to(tl.float32)
        x = x1 + x2
        mean = tl.load(Mean + rows, mask=rows < M).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        accW += dy * x * rstd
        accB += dy

    if dW:
        dw = tl.sum(accW, axis=0)
        tl.store(dW + pid, dw, mask=col_mask)
    if dB:
        db = tl.sum(accB, axis=0)
        tl.store(dB + pid, db, mask=col_mask)


def dropout_add_layernorm_forward(
    input,
    residual,
    weight=None,
    bias=None,
    dropout_p=0.0,
    eps=1e-5,
    training=True,
):
    """Forward pass for Dropout_Add_LayerNorm"""
    # For now, we ignore dropout in the implementation
    # (dropout_p is kept for API compatibility)
    M = input.shape[0]
    N = input.numel() // M

    input = input.contiguous()
    residual = residual.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)
    mean = torch.empty(M, dtype=torch.float32, device=input.device)
    rstd = torch.empty(M, dtype=torch.float32, device=input.device)

    TILE_N = triton.next_power_of_2(N)
    grid = (M, 1, 1)

    with torch_device_fn.device(input.device):
        dropout_add_layernorm_kernel[grid](
            input,
            residual,
            y,
            weight,
            bias,
            mean,
            rstd,
            M,
            N,
            eps,
            TILE_N,
        )

    return y, mean, rstd


def dropout_add_layernorm_backward(
    grad_output,
    input,
    residual,
    mean,
    rstd,
    weight=None,
    bias=None,
    dropout_p=0.0,
    eps=1e-5,
    training=True,
):
    """Backward pass for Dropout_Add_LayerNorm"""
    grad_output = grad_output.contiguous()
    input = input.contiguous()
    residual = residual.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()

    M = input.shape[0]
    N = input.numel() // M

    # Gradient for input and residual
    grad_input = torch.empty_like(input)
    grad_residual = torch.empty_like(residual)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)

    with torch_device_fn.device(input.device):
        dropout_add_layernorm_backward_kernel[grid](
            grad_output,
            input,
            residual,
            mean,
            rstd,
            weight,
            grad_input,
            grad_residual,
            M,
            N,
        )

    # Weight/bias gradient
    grid_w = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
    weight_grad = torch.empty_like(weight) if weight is not None else None
    bias_grad = torch.empty_like(bias) if bias is not None else None

    if weight is not None or bias is not None:
        with torch_device_fn.device(input.device):
            weight_bias_backward_kernel[grid_w](
                grad_output,
                input,
                residual,
                mean,
                rstd,
                weight_grad,
                bias_grad,
                M,
                N,
            )

    return grad_input, grad_residual, weight_grad, bias_grad


class DropoutAddLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, weight=None, bias=None, dropout_p=0.0, eps=1e-5):
        y, mean, rstd = dropout_add_layernorm_forward(
            input, residual, weight, bias, dropout_p, eps, input.requires_grad
        )
        ctx.save_for_backward(input, residual, mean, rstd, weight)
        ctx.dropout_p = dropout_p
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        input, residual, mean, rstd, weight = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        eps = ctx.eps

        grad_input, grad_residual, grad_weight, grad_bias = dropout_add_layernorm_backward(
            dy, input, residual, mean, rstd, weight, None, dropout_p, eps, True
        )

        return grad_input, grad_residual, grad_weight, grad_bias, None, None


def dropout_add_layernorm(input, residual, weight=None, bias=None, dropout_p=0.0, eps=1e-5):
    """Fused Add + LayerNorm operator.

    This operator fuses two operations:
    1. Add: input + residual
    2. LayerNorm: apply layer normalization

    Args:
        input: Input tensor
        residual: Residual tensor to add
        weight: LayerNorm weight (optional)
        bias: LayerNorm bias (optional)
        dropout_p: Dropout probability (default: 0.0, currently not used)
        eps: LayerNorm epsilon (default: 1e-5)

    Returns:
        Output tensor after Add -> LayerNorm
    """
    logger.debug("GEMS DROPOUT_ADD_LAYERNORM")
    return DropoutAddLayerNorm.apply(input, residual, weight, bias, dropout_p, eps)