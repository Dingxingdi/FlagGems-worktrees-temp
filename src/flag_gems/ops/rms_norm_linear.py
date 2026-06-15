import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 512}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 2048}, num_warps=4, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit(do_not_specialize=["eps"])
def rms_norm_linear_kernel(
    out_ptr,
    x_ptr,
    w_rms_ptr,
    w_linear_ptr,
    b_linear_ptr,
    M,
    N,
    K,
    eps,
    BLOCK_N: tl.constexpr,
):
    """
    Fused RMSNorm + Linear kernel.
    Each program processes one row and computes all K output elements.
    """
    # Get program ID (one per row)
    row_pid = tle.program_id(0)

    if row_pid >= M:
        return

    # ===== Step 1: Compute RMSNorm =====
    # Compute sum(x^2) for RMS computation
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # First pass: compute sum of squares
    for start_n in range(0, N, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        mask = n_offsets < N

        x = tl.load(x_ptr + row_pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        acc += x * x

    var = tl.sum(acc, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    # Second pass: compute all K output elements for this row
    # Each output element is computed sequentially
    for col in range(K):
        output = 0.0

        for start_n in range(0, N, BLOCK_N):
            n_offsets = start_n + tl.arange(0, BLOCK_N)
            mask = n_offsets < N

            # Load and normalize (in float32)
            x = tl.load(x_ptr + row_pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            w_rms = tl.load(w_rms_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            normalized = x * rrms * w_rms

            # Compute contribution to linear output: normalized @ W[col, :]
            w = tl.load(w_linear_ptr + col * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
            output = output + tl.sum(normalized * w, axis=0)

        # Add bias (in float32)
        bias = tl.load(b_linear_ptr + col).to(tl.float32)
        output = output + bias

        # Store output (convert back to original dtype)
        output = output.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_pid * K + col, output)


def rms_norm_linear(
    x, normalized_shape, rms_weight, linear_weight, bias=None, eps=1e-5
):
    """
    Fused RMSNorm + Linear operation.
    """
    logger.debug(
        "GEMS RMS_NORM_LINEAR FORWARD, [input shape]: %s, [normalized_shape]: %s, "
        "[linear_weight shape]: %s",
        x.size(),
        normalized_shape,
        linear_weight.shape,
    )

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    K = linear_weight.shape[0]

    # Handle the case where x has more than 2 dimensions
    original_shape = list(x.shape)
    if x.ndim > 2:
        x = x.view(-1, N).contiguous()
        M = x.shape[0]

    rms_weight = rms_weight.contiguous()
    linear_weight = linear_weight.contiguous()

    # Output shape: (M, K)
    output = torch.empty((M, K), dtype=x.dtype, device=x.device)

    # Handle bias - use zeros if not provided
    if bias is None:
        bias = torch.zeros(K, dtype=x.dtype, device=x.device)
    bias = bias.contiguous()

    # Grid: (M rows)
    grid = (M,)

    with torch_device_fn.device(x.device):
        rms_norm_linear_kernel[grid](
            output,
            x,
            rms_weight,
            linear_weight,
            bias,
            M,
            N,
            K,
            eps,
        )

    # Reshape output to original batch shape + out_features
    output_shape = original_shape[:-1] + [K]
    return output.view(output_shape)