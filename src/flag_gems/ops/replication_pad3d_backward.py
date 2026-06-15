import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("replication_pad3d_backward"),
    key=["H_out", "W_out"],
)
@triton.jit
def replication_pad3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_f,
    pad_t,
    pad_l,
    stride_go_n,
    stride_go_c,
    stride_go_d,
    stride_go_h,
    stride_go_w,
    stride_gi_n,
    stride_gi_c,
    stride_gi_d,
    stride_gi_h,
    stride_gi_w,
    C,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_ncd = tl.program_id(2)

    d_out_idx = pid_ncd % D_out
    nc_idx = pid_ncd // D_out
    c_idx = nc_idx % C
    n_idx = nc_idx // C

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    # Compute clamped input indices using the same logic as forward
    d_in_idx = d_out_idx - pad_f
    d_in_idx = tl.where(d_in_idx < 0, 0, d_in_idx)
    d_in_idx = tl.where(d_in_idx > D_in - 1, D_in - 1, d_in_idx)

    h_in_idx = offs_h - pad_t
    h_in_idx = tl.where(h_in_idx < 0, 0, h_in_idx)
    h_in_idx = tl.where(h_in_idx > H_in - 1, H_in - 1, h_in_idx)

    w_in_idx = offs_w - pad_l
    w_in_idx = tl.where(w_in_idx < 0, 0, w_in_idx)
    w_in_idx = tl.where(w_in_idx > W_in - 1, W_in - 1, w_in_idx)

    # Create 2D mask for valid output positions
    h_mask = offs_h < H_out
    w_mask = offs_w < W_out

    # Load gradients from output (convert to float32 for accumulation)
    go_offsets = (
        n_idx * stride_go_n
        + c_idx * stride_go_c
        + d_out_idx * stride_go_d
        + offs_h[:, None] * stride_go_h
        + offs_w[None, :] * stride_go_w
    )
    go_mask = h_mask[:, None] & w_mask[None, :]
    grad = tl.load(grad_output_ptr + go_offsets, mask=go_mask, other=0.0).to(tl.float32)

    # Store gradient to input using atomic_add (accumulate at clamped positions)
    gi_offsets = (
        n_idx * stride_gi_n
        + c_idx * stride_gi_c
        + d_in_idx * stride_gi_d
        + h_in_idx[:, None] * stride_gi_h
        + w_in_idx[None, :] * stride_gi_w
    )
    gi_mask = h_mask[:, None] & w_mask[None, :]

    tl.atomic_add(grad_input_ptr + gi_offsets, grad, mask=gi_mask, sem="relaxed")


def replication_pad3d_backward(grad_output, self, padding):
    logger.debug("GEMS replication_pad3d_backward")
    if isinstance(padding, int):
        pad_l = pad_r = pad_t = pad_b = pad_f = pad_ba = padding
    else:
        pad_l, pad_r, pad_t, pad_b, pad_f, pad_ba = padding

    N, C, D_in, H_in, W_in = self.shape
    _, _, D_out, H_out, W_out = grad_output.shape

    # Use float32 for accumulation to avoid precision issues with atomic_add
    grad_input = torch.zeros(
        (N, C, D_in, H_in, W_in), device=grad_output.device, dtype=torch.float32
    )

    grid = lambda META: (
        triton.cdiv(W_out, META["BLOCK_W"]),
        triton.cdiv(H_out, META["BLOCK_H"]),
        N * C * D_out,
    )

    replication_pad3d_backward_kernel[grid](
        grad_output,
        grad_input,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        pad_f,
        pad_t,
        pad_l,
        *grad_output.stride(),
        *grad_input.stride(),
        C,
    )

    return grad_input.to(grad_output.dtype)