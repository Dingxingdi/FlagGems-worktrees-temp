import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cumsum import cumsum
from flag_gems.ops.sort import sort
from flag_gems.ops.softmax import softmax
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset, uniform

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["n_samples", "philox_seed", "philox_offset"])
def top_p_sampling_kernel(
    sorted_probs_ptr,
    indices_ptr,
    cumsum_ptr,
    cutoff_cumsum_ptr,
    output_ptr,
    K,
    n_samples,
    philox_seed,
    philox_offset,
    BLOCK_SIZE: tl.constexpr = 128,
):
    """
    Top-P (Nucleus) sampling kernel.
    For each distribution (row), samples from the truncated distribution
    (only elements up to cutoff where cumsum <= top_p).

    Args:
        sorted_probs_ptr: sorted probabilities (descending)
        indices_ptr: original indices of sorted elements
        cumsum_ptr: cumulative probabilities
        cutoff_cumsum_ptr: cumulative probability at cutoff for each row
        output_ptr: output sampled indices
        K: number of vocabulary items
        n_samples: number of samples per distribution
    """
    row_id = tl.program_id(0)
    sample_id = tl.program_id(1)

    row_offset = row_id * K
    sorted_probs_ptr += row_offset
    indices_ptr += row_offset
    cumsum_ptr += row_offset
    output_ptr += row_id * n_samples + sample_id

    # Load cumulative probability at cutoff for this row
    cutoff_cumsum = tl.load(cutoff_cumsum_ptr + row_id)

    # Generate random number for sampling
    random_offset = row_id * n_samples + sample_id
    rv, _, _, _ = uniform(philox_seed, philox_offset, random_offset)
    rv += 0.0001
    rv = tl.where(rv > 0.9999, 0.9999, rv)

    # Scale rv by cutoff cumulative probability
    rv_scaled = rv * cutoff_cumsum

    # Binary search to find sampled index within cutoff
    start = 0
    end = K
    for _ in range(16):  # enough iterations for K up to 65536
        mid = (start + end) // 2
        mid_cumsum = tl.load(cumsum_ptr + mid)
        start = tl.where(mid_cumsum < rv_scaled, mid + 1, start)
        end = tl.where(mid_cumsum < rv_scaled, end, mid)

    sampled_idx = start
    # sampled_idx = tl.where(sampled_idx >= K, K - 1, sampled_idx)

    # Get original vocabulary index
    original_idx = tl.load(indices_ptr + sampled_idx)

    tl.store(output_ptr, original_idx)


def _compute_cutoff(cumsum_probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Compute the cutoff index for each row.
    cutoff is the first index where cumsum > top_p, or 1 if none exceed.
    """
    # Find positions where cumsum exceeds top_p
    exceeds = cumsum_probs > top_p
    # Get the first such position for each row
    cutoff_indices = torch.argmax(exceeds.int(), dim=-1)
    # If no position exceeds top_p, use the full length
    all_below = ~exceeds.any(dim=-1)
    cutoff_indices = torch.where(all_below, cumsum_probs.size(-1), cutoff_indices)
    # Ensure at least 1 element is kept
    cutoff_indices = torch.maximum(cutoff_indices, torch.ones_like(cutoff_indices))
    return cutoff_indices


def _get_cutoff_cumsum(cumsum_probs: torch.Tensor, cutoff_indices: torch.Tensor) -> torch.Tensor:
    """
    Get the cumulative probability value at each cutoff index.
    """
    batch_size = cumsum_probs.size(0)
    row_indices = torch.arange(batch_size, device=cumsum_probs.device)
    col_indices = cutoff_indices - 1  # Get cumsum at cutoff-1 (last included element)
    cutoff_cumsum = cumsum_probs[row_indices, col_indices]
    return cutoff_cumsum


def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0,
                   n_samples: int = 1, generator: torch.Generator = None) -> torch.Tensor:
    """
    Top-P (Nucleus) sampling from a distribution.

    Args:
        logits: Input tensor of shape (batch_size, vocab_size) or (vocab_size,)
        top_p: Cumulative probability threshold for nucleus sampling (default: 0.9)
        temperature: Temperature for softmax (default: 1.0)
        n_samples: Number of samples to draw (default: 1)
        generator: Optional random generator

    Returns:
        Sampled indices of shape (batch_size, n_samples) or (n_samples,)
    """
    logger.debug("GEMS TOP_P_SAMPLING")

    # Validate inputs
    assert logits.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64), \
        f"Unsupported dtype: {logits.dtype}"
    assert 0 < top_p <= 1.0, f"top_p must be in (0, 1], got {top_p}"
    assert temperature > 0, f"temperature must be positive, got {temperature}"

    # Handle input shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        is_1d = True
    else:
        is_1d = False

    batch_size = logits.size(0)
    vocab_size = logits.size(1)

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert logits to probabilities via softmax
    probs = softmax(logits, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = sort(probs, dim=-1, descending=True)

    # Compute cumulative probabilities using FlagGems cumsum
    cumsum_probs = cumsum(sorted_probs, dim=-1)

    # Compute cutoff indices for each row
    cutoff_indices = _compute_cutoff(cumsum_probs, top_p)

    # Get cumulative probability at each cutoff
    cutoff_cumsum = _get_cutoff_cumsum(cumsum_probs, cutoff_indices)

    # For each row, sample from the truncated distribution
    output = torch.empty((batch_size, n_samples), dtype=torch.long, device=logits.device)

    philox_seed, philox_offset = philox_backend_seed_offset(batch_size * n_samples, generator=generator)

    grid = (batch_size, n_samples)
    BLOCK_SIZE = min(128, triton.next_power_of_2(vocab_size))

    top_p_sampling_kernel[grid](
        sorted_probs,
        sorted_indices,
        cumsum_probs,
        cutoff_cumsum,
        output,
        vocab_size,
        n_samples,
        philox_seed,
        philox_offset,
        BLOCK_SIZE,
    )

    # Handle output shape
    if n_samples == 1:
        output = output.squeeze(-1)  # (batch_size, 1) -> (batch_size,)

    if is_1d:
        output = output.squeeze(0)  # (1, batch_size) -> (batch_size,)

    return output