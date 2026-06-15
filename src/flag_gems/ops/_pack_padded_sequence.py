import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _pack_padded_sequence(input_tensor: torch.Tensor, lengths: torch.Tensor, batch_first: bool):
    """
    Pack a padded sequence for efficient RNN processing.

    Args:
        input: Padded tensor of shape (batch, seq, features) if batch_first=True,
               or (seq, batch, features) if batch_first=False
        lengths: 1D tensor of shape (batch,) containing sequence lengths, sorted in descending order
        batch_first: Whether batch dimension is first

    Returns:
        (packed_sequence, batch_sizes) where:
        - packed_sequence: (total_elements, features)
        - batch_sizes: (max_seq_len,) - number of valid elements at each timestep
    """
    logger.debug("GEMS _pack_padded_sequence")

    if batch_first:
        # input shape: (batch, seq, features)
        batch_size, max_seq_len, features = input_tensor.shape
    else:
        # input shape: (seq, batch, features)
        max_seq_len, batch_size, features = input_tensor.shape

    # Compute batch_sizes on CPU
    lengths_cpu = lengths.cpu()
    batch_sizes = torch.zeros(max_seq_len, dtype=torch.int64, device='cpu')
    for t in range(max_seq_len):
        batch_sizes[t] = (lengths_cpu > t).sum().item()

    # Trim trailing zeros from batch_sizes
    while len(batch_sizes) > 0 and batch_sizes[-1] == 0:
        batch_sizes = batch_sizes[:-1]

    # Update max_seq_len after trimming
    max_seq_len = len(batch_sizes)

    total_elements = batch_sizes.sum().item()

    if total_elements == 0:
        # All sequences are empty
        return (
            torch.empty((0, features), dtype=input_tensor.dtype, device=input_tensor.device),
            batch_sizes.to(input_tensor.device),
        )

    # Compute cumulative sum of batch_sizes
    batch_sizes_cumsum = torch.cumsum(batch_sizes, dim=0)

    # Create output tensor
    output = torch.empty((total_elements, features), dtype=input_tensor.dtype, device=input_tensor.device)

    # Use Python loop to fill output - this is correct and simple
    # Output format: all batch elements at timestep 0, then all at timestep 1, etc.
    out_idx = 0
    for timestep in range(max_seq_len):
        for batch_idx in range(batch_size):
            if lengths_cpu[batch_idx] > timestep:
                # This element is valid
                if batch_first:
                    output[out_idx] = input_tensor[batch_idx, timestep, :]
                else:
                    output[out_idx] = input_tensor[timestep, batch_idx, :]
                out_idx += 1

    return output, batch_sizes.to(input_tensor.device)