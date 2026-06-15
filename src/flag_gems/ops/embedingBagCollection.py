import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def embedingBagCollection_kernel(
    output_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    max_indices_ptr,
    weight_ptr,
    indices_ptr,
    offsets_ptr,
    weight_stride0,  # num_embeddings
    weight_stride1,  # embedding_dim
    indices_numel,  # total number of indices
    num_bags,
    embedding_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for embedding bag collection operation.

    Args:
        output_ptr: output tensor (num_bags, embedding_dim)
        offset2bag_ptr: mapping from index to bag (indices_numel,)
        bag_size_ptr: size of each bag (num_bags,)
        max_indices_ptr: max index in each bag (num_bags,)
        weight_ptr: embedding table (num_embeddings, embedding_dim)
        indices_ptr: indices into embedding table (total_indices,)
        offsets_ptr: starting position of each bag (num_bags + 1,)
    """
    pid = tle.program_id(0)

    # Each program handles one bag
    if pid >= num_bags:
        return

    # Get bag range
    bag_start = tl.load(offsets_ptr + pid).to(tl.int32)
    bag_end = tl.load(offsets_ptr + pid + 1).to(tl.int32) if pid < num_bags - 1 else indices_numel

    bag_size = bag_end - bag_start
    tl.store(bag_size_ptr + pid, bag_size)

    # Initialize output for this bag
    output_row_start = pid * embedding_dim
    max_idx = -1

    # Process each index in the bag and accumulate to output directly
    for i in range(bag_size):
        indices_idx = bag_start + i
        embedding_idx = tl.load(indices_ptr + indices_idx).to(tl.int32)

        # Store offset2bag mapping
        tl.store(offset2bag_ptr + indices_idx, pid)

        # Track max index
        if embedding_idx > max_idx:
            max_idx = embedding_idx

        # Load embedding and accumulate directly to output
        embedding_start = embedding_idx * embedding_dim

        # Process in blocks to handle arbitrary embedding_dim
        for dim in range(0, embedding_dim, BLOCK_SIZE):
            dim_start = dim
            dim_end = tl.minimum(dim + BLOCK_SIZE, embedding_dim)
            dim_range = tl.arange(0, BLOCK_SIZE)
            dim_mask = dim_range < (dim_end - dim_start)

            # Load the embedding row
            embedding_vals = tl.load(
                weight_ptr + embedding_start + dim + dim_range,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)

            # Load current output and add
            current_vals = tl.load(
                output_ptr + output_row_start + dim + dim_range,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)

            new_vals = current_vals + embedding_vals

            tl.store(
                output_ptr + output_row_start + dim + dim_range,
                new_vals,
                mask=dim_mask,
            )

    # Store max indices
    tl.store(max_indices_ptr + pid, max_idx)


def embedingBagCollection(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,  # 0=sum, 1=mean, 2=max
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    """Embedding bag collection operation.

    Computes sums or means of slices of an embedding table.

    Args:
        weight: Embedding table (num_embeddings, embedding_dim)
        indices: Indices into embedding table (total_indices,)
        offsets: Starting position of each bag (num_bags + 1,)
        scale_grad_by_freq: Whether to scale gradients by frequency
        mode: 0=sum, 1=mean, 2=max
        sparse: Whether to use sparse gradient
        per_sample_weights: Weights for each sample
        include_last_offset: Whether offsets includes last offset
        padding_idx: Padding index to ignore

    Returns:
        Tuple of (output, offset2bag, bag_size, maximum_indices)
    """
    logger.debug("GEMS embedingBagCollection FORWARD")

    # Ensure contiguous
    weight = weight.contiguous()
    indices = indices.contiguous()
    offsets = offsets.contiguous()

    num_embeddings = weight.shape[0]
    embedding_dim = weight.shape[1]
    indices_numel = indices.numel()

    # Calculate num_bags
    num_bags = offsets.shape[0]
    if include_last_offset:
        num_bags = num_bags - 1

    # Allocate outputs - initialized to zero for accumulation
    output = torch.zeros(
        (num_bags, embedding_dim),
        device=weight.device,
        dtype=weight.dtype,
    )
    offset2bag = torch.zeros((indices_numel,), device=weight.device, dtype=torch.long)
    bag_size = torch.zeros((num_bags,), device=weight.device, dtype=torch.long)
    max_indices = torch.zeros((num_bags,), device=weight.device, dtype=torch.long)

    # Calculate grid
    # Use num_bags as the grid dimension, each block processes one bag
    grid = (num_bags,)

    # BLOCK_SIZE for processing embedding dimensions
    BLOCK_SIZE = triton.next_power_of_2(embedding_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Cap at 1024

    with torch_device_fn.device(weight.device):
        embedingBagCollection_kernel[grid](
            output,
            offset2bag,
            bag_size,
            max_indices,
            weight,
            indices,
            offsets,
            weight.stride(0),
            weight.stride(1),
            indices_numel,
            num_bags,
            embedding_dim,
            BLOCK_SIZE,
        )

    # Handle mode (mean)
    if mode == 1 and per_sample_weights is None:
        # Mean mode: divide by bag size
        bag_size_float = bag_size.to(torch.float32).reshape(-1, 1)
        output = output / bag_size_float

    return output, offset2bag, bag_size, max_indices