import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _sparse_semi_structured_linear_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_b,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for sparse semi-structured linear.
    This implementation uses dense matrix multiplication as a fallback
    since the sparse semi-structured format requires specific hardware support.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m * BLOCK_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load input and weight
    a_ptrs = input_ptr + (offs_am[:, None] * stride_im + offs_k[None, :] * stride_ik)
    b_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ik
        b_ptrs += BLOCK_SIZE_K * stride_wk

    # Convert accumulator to float32 first (accumulator is always float32)
    c = accumulator.to(tl.float32)

    # Add bias if present
    if has_bias:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.0)
        c = c + bias[None, :]

    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _sparse_semi_structured_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    meta: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Sparse semi-structured linear layer.

    This implementation provides a dense fallback for hardware that doesn't
    support the sparse semi-structured format.

    Args:
        input: Input tensor of shape (..., K)
        weight: Weight tensor of shape (N, K)
        meta: Metadata tensor for sparsity pattern (K/2, K/4)
        bias: Optional bias tensor of shape (N,)
        activation: Optional activation function ("relu", "silu", "gelu")
        out_dtype: Optional output dtype

    Returns:
        Output tensor of shape (..., N)
    """
    logger.debug("GEMS _sparse_semi_structured_linear")

    # Handle 1D input
    squeeze_output = False
    if input.ndim == 1:
        input = input.unsqueeze(0)
        squeeze_output = True

    # Handle batch dimensions
    *batch_dims, K = input.shape
    M = 1
    for dim in batch_dims:
        M *= dim

    input_flat = input.view(M, K)
    N = weight.shape[0]

    # Determine output dtype
    if out_dtype is not None:
        output_dtype = out_dtype
    else:
        output_dtype = input.dtype

    # Allocate output as float32 for computation
    output_shape = (*batch_dims, N)
    output_flat = torch.empty((M, N), dtype=torch.float32, device=input.device)

    # Kernel configuration
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Cast to float32 for computation if needed
    input_compute = input_flat.to(torch.float32) if input.dtype in (torch.float16, torch.bfloat16) else input_flat
    weight_compute = weight.to(torch.float32) if weight.dtype in (torch.float16, torch.bfloat16) else weight
    bias_compute = None
    if bias is not None:
        bias_compute = bias.to(torch.float32) if bias.dtype in (torch.float16, torch.bfloat16) else bias

    has_bias = bias is not None

    _sparse_semi_structured_linear_kernel[grid](
        input_compute,
        weight_compute,
        output_flat,
        bias_compute if bias_compute is not None else 0,
        M,
        N,
        K,
        input_compute.stride(0),
        input_compute.stride(1),
        weight_compute.stride(0),  # stride_wn - N dimension
        weight_compute.stride(1),  # stride_wk - K dimension
        output_flat.stride(0),
        output_flat.stride(1),
        0 if bias_compute is None else bias_compute.stride(0),
        has_bias,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Convert output to target dtype
    output = output_flat.to(output_dtype).view(output_shape)

    # Apply activation if specified
    if activation is not None:
        if activation == "relu":
            output = torch.nn.functional.relu(output)
        elif activation == "silu" or activation == "swish":
            output = torch.nn.functional.silu(output)
        elif activation == "gelu":
            output = torch.nn.functional.gelu(output)
        elif activation != "none":
            raise ValueError(f"Unsupported activation: {activation}")

    if squeeze_output:
        output = output.squeeze(0)

    return output