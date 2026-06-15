import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def topk_softmax_sample_kernel(
    topk_values_ptr,
    topk_indices_ptr,
    output_ptr,
    K: tl.constexpr,
    B: tl.constexpr,
    seed: tl.constexpr,
    offset: tl.constexpr,
):
    """
    Triton kernel for top-k sampling:
    1. Apply softmax to top-k values
    2. Compute cumulative sum
    3. Sample from the distribution using inverse CDF method

    The kernel processes one batch element per program.
    """
    batch_idx = tl.program_id(0)

    # Generate random number for sampling using tl.rand
    # This generates a random float in [0, 1)
    rng = tl.rand(seed, offset + batch_idx)
    rv = rng.to(tl.float32)

    # Add small perturbation to avoid exact zero
    rv = rv + 1e-6

    # Load top-k values and compute softmax
    topk_base = batch_idx * K

    # Find max for numerical stability (cast to fp32 for consistency)
    max_val = tl.load(topk_values_ptr + topk_base + 0).to(tl.float32)
    for j in range(1, K):
        val = tl.load(topk_values_ptr + topk_base + j).to(tl.float32)
        max_val = tl.maximum(max_val, val)

    # Compute sum of exp values (all in fp32)
    sum_exp = 0.0
    for j in range(K):
        val = tl.load(topk_values_ptr + topk_base + j).to(tl.float32)
        sum_exp = sum_exp + tl.exp(val - max_val)

    # Sample using inverse CDF
    cumsum = 0.0
    sampled_idx = K - 1  # Default to last index

    for j in range(K):
        val = tl.load(topk_values_ptr + topk_base + j).to(tl.float32)
        prob = tl.exp(val - max_val) / sum_exp
        cumsum = cumsum + prob
        # Update sampled index if random value is within cumulative probability
        sampled_idx = tl.where(rv <= cumsum, j, sampled_idx)

    # Get the original index from top-k indices
    original_idx = tl.load(topk_indices_ptr + topk_base + sampled_idx)

    # Store result
    tl.store(output_ptr + batch_idx, original_idx)


def top_k_sampling(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Performs top-k sampling on the given logits.

    This function:
    1. Takes the top-k largest values from the last dimension
    2. Applies softmax to normalize them to a probability distribution
    3. Samples from this distribution using Triton kernels

    Args:
        logits: Input tensor of shape (batch_size, num_tokens) or (num_tokens,)
        k: Number of top tokens to keep
        dim: Dimension to operate on (default: -1)

    Returns:
        sampled_indices: Tensor of sampled indices
    """
    logger.debug("GEMS Top_K_Sampling")

    # Handle input shapes
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = logits.shape[0]
    num_tokens = logits.shape[1]

    # Clamp k to valid range
    k = min(k, num_tokens)

    # Get top-k values and indices using torch.topk
    # This is a preprocessing step - the actual sampling is done in Triton
    topk_values, topk_indices = torch.topk(logits, k, dim=1, largest=True, sorted=True)

    # Allocate output tensor
    sampled_indices = torch.empty(batch_size, dtype=torch.int64, device=logits.device)

    # Get random seed and offset
    seed = torch.randint(2**31, (1,)).item()
    offset = torch.randint(2**31, (1,)).item()

    # Launch Triton kernel
    grid = (batch_size,)
    topk_softmax_sample_kernel[grid](
        topk_values,
        topk_indices,
        sampled_indices,
        k,
        batch_size,
        seed,
        offset,
    )

    if squeeze_output:
        sampled_indices = sampled_indices.squeeze(0)

    return sampled_indices


# Reference implementation using pure PyTorch for testing comparison
def top_k_sampling_reference(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """
    Reference implementation using pure PyTorch operations.
    Used for testing accuracy comparison.
    """
    # Handle input shapes
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = logits.shape[0]
    num_tokens = logits.shape[1]

    # Clamp k to valid range
    k = min(k, num_tokens)

    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(logits, k, dim=1, largest=True, sorted=True)

    # Apply softmax to get probabilities
    topk_values_max = topk_values.max(dim=1, keepdim=True)[0]
    topk_values_exp = torch.exp(topk_values - topk_values_max)
    topk_probs = topk_values_exp / topk_values_exp.sum(dim=1, keepdim=True)

    # Sample from the distribution
    cumsum = torch.cumsum(topk_probs, dim=1)

    # Generate random numbers
    rand = torch.rand(batch_size, device=logits.device, dtype=logits.dtype)

    # Find sampled indices
    sampled = torch.searchsorted(cumsum, rand, right=True)
    sampled = torch.clamp(sampled, 0, k - 1)

    # Map back to original indices
    sampled_indices = torch.gather(topk_indices, 1, sampled.unsqueeze(1)).squeeze(1)

    if squeeze_output:
        sampled_indices = sampled_indices.squeeze(0)

    return sampled_indices