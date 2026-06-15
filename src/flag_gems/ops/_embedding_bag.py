import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def embedding_bag_kernel(
    output_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    max_indices_ptr,
    weight_ptr,
    indices_ptr,
    offsets_ptr,
    num_embeddings,
    embedding_dim: tl.constexpr,
    num_indices: tl.constexpr,
    num_bags: tl.constexpr,
    mode: tl.constexpr,
    include_last_offset: tl.constexpr,
    padding_idx: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    embedding_bag kernel.

    Args:
        output_ptr: pointer to output tensor of shape [num_bags, embedding_dim]
        offset2bag_ptr: pointer to offset2bag tensor of shape [num_indices]
        bag_size_ptr: pointer to bag_size tensor of shape [num_bags]
        max_indices_ptr: pointer to max_indices tensor of shape [num_bags, embedding_dim] (for mode 2)
        weight_ptr: pointer to weight tensor of shape [num_embeddings, embedding_dim]
        indices_ptr: pointer to indices tensor of shape [num_indices]
        offsets_ptr: pointer to offsets tensor of shape [num_bags] or [num_bags + 1] if include_last_offset
    """

    # Get program IDs
    pid = tle.program_id(0)

    if pid >= num_bags:
        return

    # Calculate bag boundaries
    bag_start = tl.load(offsets_ptr + pid).to(tl.int64)
    if include_last_offset:
        if pid == num_bags - 1:
            bag_end = tl.cast(num_indices, tl.int64)
        else:
            bag_end = tl.load(offsets_ptr + pid + 1).to(tl.int64)
    else:
        if pid == num_bags - 1:
            bag_end = tl.cast(num_indices, tl.int64)
        else:
            bag_end = tl.load(offsets_ptr + pid + 1).to(tl.int64)

    bag_size = bag_end - bag_start

    # Compute reduction based on mode
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < embedding_dim

    if mode == 0:  # SUM
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for i in range(bag_start, bag_end):
            idx = tl.load(indices_ptr + i)
            # Check for padding_idx
            if padding_idx >= 0:
                valid = idx != padding_idx
            else:
                valid = True

            if valid:
                weight_offset = idx * embedding_dim
                embedding = tl.load(weight_ptr + weight_offset + cols, mask=mask, other=0.0)
                acc = acc + embedding.to(tl.float32)
                # Mark which bag this index belongs to
                tl.store(offset2bag_ptr + i, pid)

        if bag_size > 0:
            tl.store(output_ptr + pid * embedding_dim + cols, acc, mask=mask)
            tl.store(bag_size_ptr + pid, bag_size)
        else:
            # Empty bag - store zeros
            tl.store(output_ptr + pid * embedding_dim + cols, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
            tl.store(bag_size_ptr + pid, 0)

    elif mode == 1:  # MEAN
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for i in range(bag_start, bag_end):
            idx = tl.load(indices_ptr + i)
            if padding_idx >= 0:
                valid = idx != padding_idx
            else:
                valid = True

            if valid:
                weight_offset = idx * embedding_dim
                embedding = tl.load(weight_ptr + weight_offset + cols, mask=mask, other=0.0)
                acc = acc + embedding.to(tl.float32)
                tl.store(offset2bag_ptr + i, pid)

        if bag_size > 0:
            mean_acc = acc / tl.cast(bag_size, tl.float32)
            tl.store(output_ptr + pid * embedding_dim + cols, mean_acc, mask=mask)
            tl.store(bag_size_ptr + pid, bag_size)
        else:
            tl.store(output_ptr + pid * embedding_dim + cols, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
            tl.store(bag_size_ptr + pid, 0)

    elif mode == 2:  # MAX
        # For max mode, we need to track the max value and index per dimension
        max_val = tl.full([BLOCK_SIZE], dtype=tl.float32, value=float("-inf"))
        max_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

        for i in range(bag_start, bag_end):
            idx = tl.load(indices_ptr + i)
            if padding_idx >= 0:
                valid = idx != padding_idx
            else:
                valid = True

            if valid:
                weight_offset = idx * embedding_dim
                embedding = tl.load(weight_ptr + weight_offset + cols, mask=mask, other=0.0).to(tl.float32)
                cmp_mask = embedding > max_val
                max_val = tl.where(cmp_mask, embedding, max_val)
                max_idx = tl.where(cmp_mask, tl.cast(idx, tl.int64), max_idx)
                tl.store(offset2bag_ptr + i, pid)

        if bag_size > 0:
            tl.store(output_ptr + pid * embedding_dim + cols, max_val, mask=mask)
            tl.store(bag_size_ptr + pid, bag_size)
            # Store max indices (repeated for each embedding dimension)
            for d in range(embedding_dim):
                tl.store(max_indices_ptr + pid * embedding_dim + d, max_idx)
        else:
            tl.store(output_ptr + pid * embedding_dim + cols, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
            tl.store(bag_size_ptr + pid, 0)
            for d in range(embedding_dim):
                tl.store(max_indices_ptr + pid * embedding_dim + d, 0)


def _embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    """
    Computes bag of embeddings.

    This operator is not fully supported in FlagGems. It currently provides
    a basic implementation for the forward pass only.

    Args:
        weight: embedding table of shape [num_embeddings, embedding_dim]
        indices: indices into weight of shape [num_indices]
        offsets: offsets that specify bag boundaries
        scale_grad_by_freq: whether to scale gradients by frequency (not supported)
        mode: 0=sum, 1=mean, 2=max
        sparse: whether to use sparse representation (not supported)
        per_sample_weights: weights for each sample (not supported)
        include_last_offset: whether offsets includes the last offset
        padding_idx: padding index to ignore

    Returns:
        tuple of (output, offset2bag, bag_size, maximum_indices)
    """
    logger.debug("GEMS _EMBEDDING_BAG")

    assert not sparse, "Sparse embedding_bag is not supported"
    assert per_sample_weights is None, "per_sample_weights is not supported"

    num_embeddings, embedding_dim = weight.shape
    num_indices = indices.numel()
    num_offsets = offsets.numel()
    # When include_last_offset=True, offsets includes the final boundary, so it has num_bags + 1 elements
    # When include_last_offset=False, offsets has num_bags elements
    num_bags = num_offsets - 1 if include_last_offset else num_offsets

    # Contiguous tensors
    weight = weight.contiguous()
    indices = indices.contiguous()
    offsets = offsets.contiguous()

    # Allocate output tensors
    output = torch.zeros((num_bags, embedding_dim), device=weight.device, dtype=weight.dtype)
    offset2bag = torch.zeros(num_indices, device=weight.device, dtype=torch.int64)
    # PyTorch returns bag_size and max_indices with num_offsets elements (not num_bags)
    bag_size = torch.zeros(num_offsets, device=weight.device, dtype=torch.int64)
    max_indices = torch.zeros((num_offsets, embedding_dim), device=weight.device, dtype=torch.int64)

    BLOCK_SIZE = triton.next_power_of_2(embedding_dim)

    with torch_device_fn.device(weight.device):
        embedding_bag_kernel[(num_bags,)](
            output,
            offset2bag,
            bag_size,
            max_indices,
            weight,
            indices,
            offsets,
            num_embeddings,
            embedding_dim,
            num_indices,
            num_bags,
            mode,
            include_last_offset,
            padding_idx,
            BLOCK_SIZE,
        )

    return output, offset2bag, bag_size, max_indices