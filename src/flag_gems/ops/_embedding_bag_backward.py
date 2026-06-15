import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _embedding_bag_backward_sum_mean_kernel(
    grad_output_ptr,
    indices_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    embedding_dim,
    scale_by_bag_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Sum/Mean mode: distribute gradients to indices in each bag.

    Args:
        scale_by_bag_size: if True, divide gradient by bag_size (mean mode)
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < embedding_dim

    # Load index and its bag assignment
    idx = tl.load(indices_ptr + pid_n).to(tl.int32)
    bag_id = tl.load(offset2bag_ptr + pid_n).to(tl.int32)

    # Check if index is valid
    valid_idx = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)
    mask = mask_d & valid_idx

    # Load bag gradient
    go_ptrs = grad_output_ptr + bag_id * embedding_dim + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    # For mean mode, scale by 1/bag_size
    if scale_by_bag_size:
        bag_size = tl.load(bag_size_ptr + bag_id).to(tl.float32)
        bag_size = bag_size if bag_size > 0 else 1.0
        go = go / bag_size

    # Accumulate to grad_weight
    gw_ptrs = grad_weight_ptr + idx * embedding_dim + offs_d
    tl.atomic_add(gw_ptrs, go, mask=mask)


@triton.jit
def _embedding_bag_backward_max_kernel(
    grad_output_ptr,
    maximum_indices_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    embedding_dim,
    BLOCK_D: tl.constexpr,
):
    """Max mode: only the maximum_indices receive gradients."""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < embedding_dim

    # Load gradient for this bag and dimension block
    go_ptrs = grad_output_ptr + pid_b * embedding_dim + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    # Load the index that had the max value for each dimension
    max_idx_ptrs = maximum_indices_ptr + pid_b * embedding_dim + offs_d
    max_idx = tl.load(max_idx_ptrs, mask=mask_d, other=0).to(tl.int32)

    # Check if index is valid
    valid = (max_idx != padding_idx) & (max_idx >= 0) & (max_idx < num_weights)
    mask = mask_d & valid

    # Accumulate to grad_weight
    gw_ptrs = grad_weight_ptr + max_idx * embedding_dim + offs_d
    tl.atomic_add(gw_ptrs, go, mask=mask)


def _embedding_bag_backward(
    grad: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    offset2bag: torch.Tensor,
    bag_size: torch.Tensor,
    maximum_indices: torch.Tensor,
    num_weights: int,
    scale_grad_by_freq: bool = False,
    mode: int = 0,
    sparse: bool = False,
    per_sample_weights: torch.Tensor = None,
    padding_idx: int = -1,
):
    logger.debug("GEMS: _embedding_bag_backward")

    assert indices.dtype in (torch.int32, torch.int64), "Indices must be int32 or int64."
    assert grad.is_cuda and indices.is_cuda and grad.device == indices.device, \
        "All inputs must be CUDA tensors on the same device."

    device = grad.device
    num_bags = offsets.numel()
    if num_bags > 0:
        num_bags = num_bags - 1  # offsets has num_bags + 1 elements
    embedding_dim = grad.shape[-1]

    # Flatten grad_output to (num_bags, embedding_dim)
    go = grad.contiguous().view(-1, embedding_dim)

    # Flatten indices and offset2bag
    idx = indices.contiguous().view(-1)
    o2b = offset2bag.contiguous().view(-1)
    n_indices = idx.numel()

    # Allocate output gradient weight
    grad_weight_fp32 = torch.zeros((num_weights, embedding_dim), device=device, dtype=torch.float32)

    BLOCK_D = 128
    grid = (n_indices, triton.cdiv(embedding_dim, BLOCK_D))

    # For now, ignore scale_grad_by_freq and per_sample_weights (they require extra kernels)
    if mode == 0:
        # Sum mode
        _embedding_bag_backward_sum_mean_kernel[grid](
            go,
            idx,
            o2b,
            bag_size,
            grad_weight_fp32,
            num_weights,
            padding_idx,
            embedding_dim,
            False,  # scale_by_bag_size = False for sum
            BLOCK_D=BLOCK_D,
        )
    elif mode == 1:
        # Mean mode
        _embedding_bag_backward_sum_mean_kernel[grid](
            go,
            idx,
            o2b,
            bag_size,
            grad_weight_fp32,
            num_weights,
            padding_idx,
            embedding_dim,
            True,  # scale_by_bag_size = True for mean
            BLOCK_D=BLOCK_D,
        )
    elif mode == 2:
        # Max mode
        max_indices = maximum_indices.contiguous().view(-1, embedding_dim)
        grid_max = (num_bags, triton.cdiv(embedding_dim, BLOCK_D))
        _embedding_bag_backward_max_kernel[grid_max](
            go,
            max_indices,
            grad_weight_fp32,
            num_weights,
            padding_idx,
            embedding_dim,
            BLOCK_D=BLOCK_D,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if grad.dtype != torch.float32:
        return grad_weight_fp32.to(grad.dtype)
    return grad_weight_fp32