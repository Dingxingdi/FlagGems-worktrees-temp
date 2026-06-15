import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


# Autotune configs for matmul (similar to addmm)
@libentry()
@libtuner(
    configs=runtime.ops_get_configs("addmm", pre_hook=None)
    if os.environ.get("USE_FLAGTUNE") == "1"
    else runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
    strategy=runtime.get_expand_config("addmm")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def matmul_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Batch ID
    pid_b = tl.program_id(2)

    # Offset for batch
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_ptrs = O + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    o_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(o_ptrs, accumulator.to(O.dtype.element_ty), mask=o_mask)


def matmul(input, other):
    logger.debug("GEMS MATMUL")

    input_dim = input.dim()
    other_dim = other.dim()

    # Handle 1D x 1D case (dot product)
    if input_dim == 1 and other_dim == 1:
        assert input.shape == other.shape, "Input vectors must have the same shape"
        return torch.dot(input, other)

    # Handle 1D x 2D case
    # Prepend 1 to input's dimension for matrix multiply, then remove it
    if input_dim == 1 and other_dim == 2:
        input = input.unsqueeze(0)
        result = torch.matmul(input, other)
        return result.squeeze(0)

    # Handle 2D x 1D case (matrix-vector product)
    if input_dim == 2 and other_dim == 1:
        # For matrix-vector: treat other as column vector
        other = other.unsqueeze(-1)
        result = torch.matmul(input, other)
        return result.squeeze(-1)

    # Handle N-D cases (N >= 2)
    # Need to handle the broadcasting of batch dimensions
    input_shape = input.shape
    other_shape = other.shape

    # Get the matrix dimensions (last two dimensions)
    if input_dim >= 2 and other_dim >= 2:
        K = input_shape[-1]
        assert K == other_shape[-2], f"Incompatible dimensions for matmul: {input_shape[-1]} vs {other_shape[-2]}"

        M = input_shape[-2]
        N = other_shape[-1]

        # Compute batch dimensions
        input_batch = input_shape[:-2] if input_dim > 2 else ()
        other_batch = other_shape[:-2] if other_dim > 2 else ()

        # Broadcast batch dimensions
        batch_dims = []
        max_batch = max(len(input_batch), len(other_batch))
        for i in range(max_batch):
            dim_i = len(input_batch) - max_batch + i
            dim_j = len(other_batch) - max_batch + i
            batch_i = input_batch[dim_i] if dim_i >= 0 else 1
            batch_j = other_batch[dim_j] if dim_j >= 0 else 1
            assert batch_i == batch_j or batch_i == 1 or batch_j == 1, \
                f"Batch dimensions {input_batch} and {other_batch} are not broadcastable"
            batch_dims.append(max(batch_i, batch_j))

        # Total number of batch elements
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

        # Reshape inputs to (batch, M, K) and (batch, K, N)
        input_reshaped = input.reshape(batch_size, M, K).contiguous()
        other_reshaped = other.reshape(batch_size, K, N).contiguous()

        # Allocate output
        output = torch.empty((batch_size, M, N), dtype=input.dtype, device=input.device)

        def grid_fn(meta):
            return (
                triton.cdiv(M, meta["BLOCK_SIZE_M"]),
                triton.cdiv(N, meta["BLOCK_SIZE_N"]),
                batch_size,
            )

        with torch_device_fn.device(input.device):
            matmul_kernel[grid_fn](
                input_reshaped,
                other_reshaped,
                output,
                M,
                N,
                K,
                input_reshaped.stride(0),
                input_reshaped.stride(1),
                input_reshaped.stride(2),
                other_reshaped.stride(0),
                other_reshaped.stride(1),
                other_reshaped.stride(2),
                output.stride(0),
                output.stride(1),
                output.stride(2),
            )

        # Reshape output back to original batch dimensions + (M, N)
        output = output.reshape(*batch_dims, M, N)

        # For 2D x 2D case, squeeze batch dimensions
        if input_dim == 2 and other_dim == 2:
            output = output.squeeze(0) if output.dim() > 2 else output

        return output

    # Fallback to torch for unsupported cases
    return torch.matmul(input, other)