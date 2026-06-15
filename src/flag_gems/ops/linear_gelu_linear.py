import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)

# Get cross-backend compatible functions
tanh = tl_extra_shim.tanh
pow = tl_extra_shim.pow


@libentry()
@triton.jit(do_not_specialize=["has_bias1"])
def linear_gelu_linear_kernel(
    input_ptr,
    weight1_ptr,
    bias1_ptr,
    weight2_ptr,
    bias2_ptr,
    output_ptr,
    M,
    N,  # intermediate dimension
    K,  # hidden dimension
    stride_im,
    stride_ik,
    stride_w1k,
    stride_w1n,
    stride_w2n,
    stride_w2o,
    stride_om,
    stride_oo,
    has_bias1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused Linear + GeLU kernel.

    This kernel computes: intermediate = gelu(input @ weight1 + bias1)
    Then the second matmul is done separately.

    Dimensions:
        input: (M, K) - M is batch, K is hidden dim
        weight1: (K, N) - K is hidden dim, N is intermediate dim
        intermediate: (M, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create pointers for input and weight1
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    weight1_ptrs = weight1_ptr + (offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n)

    # Accumulator for first matmul
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K loop for first matmul
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load input block
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(input_ptrs, mask=input_mask, other=0.0).to(tl.float32)

        # Load weight1 block
        weight_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        b = tl.load(weight1_ptrs, mask=weight_mask, other=0.0).to(tl.float32)

        # Matrix multiply
        acc += tl.dot(a, b, allow_tf32=False)

        # Update pointers
        input_ptrs += BLOCK_SIZE_K * stride_ik
        weight1_ptrs += BLOCK_SIZE_K * stride_w1k

    # Add bias1 if present
    if has_bias1:
        bias1_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias1_mask = bias1_offs < N
        bias1 = tl.load(bias1_ptr + bias1_offs, mask=bias1_mask, other=0.0).to(tl.float32)
        acc += bias1

    # Apply GeLU activation
    # GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
    gelu_scale: tl.constexpr = 0.79788456
    gelu_out = 0.5 * acc * (1 + tanh(gelu_scale * acc * (1 + 0.044715 * pow(acc, 2))))

    # Store intermediate result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + (offs_cm[:, None] * stride_om + offs_cn[None, :] * stride_oo)
    output_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, gelu_out.to(gelu_out.dtype), mask=output_mask)


def linear_gelu_linear(
    input: torch.Tensor,
    weight1: torch.Tensor,
    bias1: Optional[torch.Tensor],
    weight2: torch.Tensor,
    bias2: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused Linear + GeLU + Linear operation.

    Computes: output = linear2(gelu(linear1(input)))

    Args:
        input: Input tensor of shape (..., hidden_dim)
        weight1: First linear layer weight of shape (hidden_dim, intermediate_dim)
        bias1: Optional first linear layer bias of shape (intermediate_dim,)
        weight2: Second linear layer weight of shape (intermediate_dim, hidden_dim)
        bias2: Optional second linear layer bias of shape (hidden_dim,)

    Returns:
        Output tensor of shape (..., hidden_dim)
    """
    logger.debug("GEMS Linear+GeLU+Linear")

    # Handle input shape - flatten batch dimensions
    original_shape = input.shape
    hidden_dim = input.shape[-1]
    batch_size = input.numel() // hidden_dim

    # Flatten input to 2D
    input_2d = input.contiguous().view(batch_size, hidden_dim)

    # Get dimensions
    M = batch_size  # batch size
    K = hidden_dim  # input hidden dimension
    N = weight1.shape[1]  # intermediate dimension
    O = weight2.shape[1]  # output hidden dimension (should equal K)

    # Validate dimensions
    assert weight1.shape[0] == K, f"weight1 shape mismatch: {weight1.shape} vs K={K}"
    assert weight2.shape[0] == N, f"weight2 shape mismatch: {weight2.shape} vs N={N}"
    assert weight2.shape[1] == O, f"weight2 output dim mismatch: {weight2.shape} vs O={O}"
    if bias1 is not None:
        assert bias1.shape[0] == N, f"bias1 shape mismatch: {bias1.shape} vs N={N}"
    if bias2 is not None:
        assert bias2.shape[0] == O, f"bias2 shape mismatch: {bias2.shape} vs O={O}"

    # Allocate intermediate buffer for first matmul + GeLU
    intermediate = torch.empty((M, N), device=input.device, dtype=torch.float32)

    # Prepare flags for kernel
    has_bias1 = bias1 is not None
    if bias1 is None:
        bias1 = torch.zeros(N, device=input.device, dtype=torch.float32)

    # Grid for first matmul + GeLU
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Launch first kernel: input @ weight1 + bias1 -> gelu -> intermediate
    linear_gelu_linear_kernel[grid](
        input_2d,
        weight1,
        bias1,
        weight2,  # Passed but not used in this kernel
        bias2 if bias2 is not None else torch.zeros(O, device=input.device),
        intermediate,
        M,
        N,
        K,
        input_2d.stride(0),
        input_2d.stride(1),
        weight1.stride(0),
        weight1.stride(1),
        weight2.stride(0),
        weight2.stride(1),
        intermediate.stride(0),
        intermediate.stride(1),
        has_bias1,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )

    # Convert intermediate to input dtype
    intermediate = intermediate.to(input.dtype)

    # Second linear: intermediate @ weight2 + bias2 (using torch matmul)
    output = torch.matmul(intermediate, weight2)
    if bias2 is not None:
        output = output + bias2

    # Reshape to original shape
    output = output.view(*original_shape[:-1], O)

    return output