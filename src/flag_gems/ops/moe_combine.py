import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["hidden_size", "num_experts"],
)
@triton.jit
def moe_combine_kernel(
    expert_outputs_ptr,
    weights_ptr,
    output_ptr,
    num_tokens,
    num_experts,
    hidden_size,
    expert_outputs_stride_token,
    expert_outputs_stride_expert,
    expert_outputs_stride_hidden,
    weights_stride_token,
    weights_stride_expert,
    output_stride_token,
    output_stride_hidden,
    output_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    MoE combine kernel: computes weighted sum of expert outputs.

    Args:
        expert_outputs: [num_tokens, num_experts, hidden_size]
        weights: [num_tokens, num_experts]
        output: [num_tokens, hidden_size]
    """
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size

    if token_idx >= num_tokens:
        return

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Get base pointer for this token's expert outputs and weights
    expert_outputs_base = expert_outputs_ptr + token_idx * expert_outputs_stride_token
    weights_base = weights_ptr + token_idx * weights_stride_token

    for expert_idx in range(num_experts):
        # Load expert output for this expert and convert to float32
        expert_ptr = (
            expert_outputs_base
            + expert_idx * expert_outputs_stride_expert
            + hidden_offsets * expert_outputs_stride_hidden
        )
        expert_data = tl.load(expert_ptr, mask=hidden_mask, other=0.0).to(tl.float32)

        # Load weight for this expert and convert to float32
        weight = tl.load(weights_base + expert_idx * weights_stride_expert).to(
            tl.float32
        )

        # Weighted sum in float32
        acc += expert_data * weight

    # Store result - convert back to original dtype
    output_ptr_pos = (
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden
    )
    if output_dtype == tl.float16:
        tl.store(output_ptr_pos, acc.to(tl.float16), mask=hidden_mask)
    elif output_dtype == tl.bfloat16:
        tl.store(output_ptr_pos, acc.to(tl.bfloat16), mask=hidden_mask)
    else:
        tl.store(output_ptr_pos, acc, mask=hidden_mask)


def moe_combine(expert_outputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Combine expert outputs using weighted sum.

    Args:
        expert_outputs: Tensor of shape [num_tokens, num_experts, hidden_size]
        weights: Tensor of shape [num_tokens, num_experts]

    Returns:
        Tensor of shape [num_tokens, hidden_size]
    """
    logger.debug("GEMS MOE_COMBINE")

    num_tokens, num_experts, hidden_size = expert_outputs.shape
    assert weights.shape == (num_tokens, num_experts), (
        f"weights shape {weights.shape} does not match "
        f"expected shape ({num_tokens}, {num_experts})"
    )

    # Ensure inputs are contiguous
    expert_outputs = expert_outputs.contiguous()
    weights = weights.contiguous()

    # Create output tensor
    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )

    # Get strides
    expert_outputs_strides = expert_outputs.stride()
    weights_strides = weights.stride()
    output_strides = output.stride()

    # Get output dtype for kernel
    if expert_outputs.dtype == torch.float16:
        output_dtype = tl.float16
    elif expert_outputs.dtype == torch.bfloat16:
        output_dtype = tl.bfloat16
    else:
        output_dtype = tl.float32

    # Launch kernel
    grid = lambda meta: (
        num_tokens,
        triton.cdiv(hidden_size, meta["BLOCK_SIZE"]),
    )

    moe_combine_kernel[grid](
        expert_outputs,
        weights,
        output,
        num_tokens,
        num_experts,
        hidden_size,
        expert_outputs_strides[0],
        expert_outputs_strides[1],
        expert_outputs_strides[2],
        weights_strides[0],
        weights_strides[1],
        output_strides[0],
        output_strides[1],
        output_dtype,
    )

    return output