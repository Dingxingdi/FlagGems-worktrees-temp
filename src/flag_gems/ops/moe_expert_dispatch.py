import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def moe_expert_dispatch_kernel(
    hidden_states_ptr,
    sorted_indices_ptr,
    output_ptr,
    num_tokens: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to reorder tokens based on sorted indices.

    Each program processes BLOCK_SIZE tokens in parallel. For each token,
    we read from the sorted position in hidden_states and write to the
    output position.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_tokens

    # Load sorted indices (these map output position -> input position)
    sorted_idx = tl.load(sorted_indices_ptr + offs, mask=mask, other=0)

    # Iterate over hidden_dim to copy all features
    # Using a loop over hidden_dim which is typically small (e.g., 4096)
    for d in range(hidden_dim):
        # Compute input pointer: sorted_idx * hidden_dim + d
        input_ptrs = hidden_states_ptr + sorted_idx * hidden_dim + d
        # Load values from sorted positions
        values = tl.load(input_ptrs, mask=mask, other=0.0)
        # Compute output pointer: offs * hidden_dim + d
        output_ptrs = output_ptr + offs * hidden_dim + d
        # Store values to output
        tl.store(output_ptrs, values, mask=mask)


def moe_expert_dispatch(hidden_states: torch.Tensor, expert_ids: torch.Tensor):
    """Dispatch tokens to expert buffers based on expert_ids.

    This function reorders tokens so that tokens routed to the same expert
    are stored contiguously, enabling efficient batched expert processing.

    The core reordering logic is implemented in Triton kernel.

    Args:
        hidden_states: Input tensor of shape [num_tokens, hidden_dim]
        expert_ids: Tensor of shape [num_tokens] with expert indices (0 to num_experts-1)

    Returns:
        Tuple of (output, sorted_indices) where:
            output: Reordered tensor of shape [num_tokens, hidden_dim]
            sorted_indices: Indices mapping output positions back to input positions
    """
    logger.debug("GEMS MoE_Expert_Dispatch")

    num_tokens, hidden_dim = hidden_states.shape

    # Sort tokens by expert_id to get contiguous expert blocks
    # This uses PyTorch's sort, but the actual data movement is done in Triton
    sorted_indices = torch.argsort(expert_ids, stable=True)
    sorted_indices = sorted_indices.to(hidden_states.device)

    # Allocate output tensor
    output = torch.empty_like(hidden_states)

    # Launch Triton kernel for the reordering
    # Each block processes 128 tokens
    BLOCK_SIZE: int = 128
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    moe_expert_dispatch_kernel[grid](
        hidden_states,
        sorted_indices,
        output,
        num_tokens,
        hidden_dim,
        BLOCK_SIZE,
    )

    return output, sorted_indices


def moe_expert_dispatch_(
    hidden_states: torch.Tensor, expert_ids: torch.Tensor
):
    """In-place version of moe_expert_dispatch.

    Args:
        hidden_states: Input tensor of shape [num_tokens, hidden_dim]
        expert_ids: Tensor of shape [num_tokens] with expert indices (0 to num_experts-1)

    Returns:
        The same tensor (modified in-place), now reordered
    """
    logger.debug("GEMS MoE_Expert_Dispatch_")

    output, _ = moe_expert_dispatch(hidden_states, expert_ids)
    hidden_states.copy_(output)
    return hidden_states