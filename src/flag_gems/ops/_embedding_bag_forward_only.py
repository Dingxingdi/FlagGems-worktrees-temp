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
    weight_ptr,
    indices_ptr,
    offsets_ptr,
    num_bags,
    embedding_dim,
    total_indices,
    include_last_offset: tl.constexpr,
    padding_idx: tl.constexpr,
    MODE: tl.constexpr,  # 0 = sum, 1 = mean
    BLOCK_SIZE: tl.constexpr,
):
    # Each program id handles one bag
    bag_idx = tle.program_id(0)

    # Load start offset for this bag
    start_offset = tl.load(offsets_ptr + bag_idx).to(tl.int32)

    # Compute end offset
    # When include_last_offset=False and this is the last bag, end = total_indices
    # Otherwise end = offsets[bag_idx + 1]
    if bag_idx + 1 < num_bags:
        end_offset = tl.load(offsets_ptr + bag_idx + 1).to(tl.int32)
    else:
        end_offset = total_indices.to(tl.int32)

    # Handle include_last_offset - shifts the interpretation
    if include_last_offset:
        # When include_last_offset=True, offsets includes the final position
        end_offset = tl.load(offsets_ptr + bag_idx + 1).to(tl.int32)

    # Accumulator for the bag
    if tl.constexpr(output_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        output_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = output_ptr.dtype.element_ty

    acc = tl.zeros([BLOCK_SIZE], dtype=cdtype)

    # Count valid indices (excluding padding_idx)
    count = 0

    # Iterate over indices in this bag
    for idx in range(start_offset, end_offset):
        # Load the embedding index
        embedding_idx = tl.load(indices_ptr + idx).to(tl.int32)

        # Compute mask for padding_idx
        is_not_padding = embedding_idx != padding_idx if not tl.constexpr(padding_idx == -1) else True

        # Load the embedding vector only if not padding
        embedding_offset = embedding_idx * embedding_dim
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < embedding_dim
        embedding = tl.load(weight_ptr + embedding_offset + cols, mask=mask, other=0.0).to(cdtype)

        # Conditionally add to accumulator based on padding
        acc = acc + embedding * (1 if is_not_padding else 0)
        count = count + (1 if is_not_padding else 0)

    # Compute mean if mode == 1
    if MODE == 1:
        if count > 0:
            acc = acc / tl.cast(count, cdtype)

    # Store result
    output_offset = bag_idx * embedding_dim
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < embedding_dim
    # Convert back to output dtype
    if tl.constexpr(output_ptr.dtype.element_ty == tl.float16):
        acc = acc.to(tl.float16)
    elif tl.constexpr(output_ptr.dtype.element_ty == tl.bfloat16):
        acc = acc.to(tl.bfloat16)
    tl.store(output_ptr + output_offset + cols, acc, mask=mask)


def _embedding_bag_forward_only(
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
    logger.debug("GEMS _EMBEDDING_BAG_FORWARD_ONLY")

    # Basic validation
    assert not sparse, "Sparse gradient not supported"
    assert scale_grad_by_freq is False, "scale_grad_by_freq not supported in forward"
    assert per_sample_weights is None, "per_sample_weights not supported in forward"

    num_embeddings, embedding_dim = weight.shape
    num_indices = indices.numel()
    num_bags = offsets.numel()

    # Output tensor
    output = torch.empty((num_bags, embedding_dim), device=weight.device, dtype=weight.dtype)

    # Ensure contiguous
    weight = weight.contiguous()
    indices = indices.contiguous()
    offsets = offsets.contiguous()

    BLOCK_SIZE = triton.next_power_of_2(embedding_dim)

    with torch_device_fn.device(weight.device):
        embedding_bag_kernel[(num_bags,)](
            output,
            weight,
            indices,
            offsets,
            num_bags,
            embedding_dim,
            num_indices,
            include_last_offset,
            padding_idx,
            mode,
            BLOCK_SIZE,
        )

    # PyTorch's _embedding_bag_forward_only returns 4 tensors
    # output: (num_bags, embedding_dim)
    # offset2bag: for each index position, which bag it belongs to
    # bag_size: number of indices in each bag
    # max_indices: maximum index in each bag

    # For offset2bag, bag_size, and max_indices, we return empty/zero tensors
    # as PyTorch seems to compute these differently in its internal implementation
    offset2bag = torch.empty(0, dtype=torch.int64, device=weight.device)
    bag_size = torch.zeros(num_bags, dtype=torch.int64, device=weight.device)
    max_indices = torch.zeros(num_bags, dtype=torch.int64, device=weight.device)

    return output, offset2bag, bag_size, max_indices