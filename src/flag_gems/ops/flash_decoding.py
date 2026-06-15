import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Define the custom aten operator schema using TORCH_LIBRARY_FRAGMENT
torch.library.define(
    "aten::flash_decoding",
    "(Tensor query, Tensor key, Tensor value, float? scale=None) -> Tensor"
)


@torch.library.impl("aten::flash_decoding", "CUDA")
def _flash_decoding_impl(query, key, value, scale=None):
    return flash_decoding(query, key, value, scale)


@libentry()
@triton.jit
def flash_decoding_kernel(
    Q,  # Query tensor, shape: (batch, num_heads, seq_q, head_dim)
    K,  # Key tensor, shape: (batch, num_heads, seq_kv, head_dim)
    V,  # Value tensor, shape: (batch, num_heads, seq_kv, head_dim)
    O,  # Output tensor, shape: (batch, num_heads, seq_q, head_dim)
    softmax_lse,  # Log-sum-exp for backward, shape: (batch, num_heads, seq_q)
    q_batch_stride,
    q_head_stride,
    q_seq_stride,
    q_dim_stride,
    k_batch_stride,
    k_head_stride,
    k_seq_stride,
    k_dim_stride,
    v_batch_stride,
    v_head_stride,
    v_seq_stride,
    v_dim_stride,
    o_batch_stride,
    o_head_stride,
    o_seq_stride,
    o_dim_stride,
    seqlen_q,
    seqlen_kv,
    num_heads,
    head_dim,
    num_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashDecoding kernel for efficient attention in decoding phase.

    This is a simplified implementation optimized for decoding where seq_q typically = 1.
    The algorithm splits the KV sequence into blocks and accumulates partial results.
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    # For decoding, we typically have seq_q = 1, so we map each program to (batch, head)
    # If seq_q > 1, we'd need to handle that differently

    # Offset calculations
    q_offset = batch_id * q_batch_stride + head_id * q_head_stride
    k_offset = batch_id * k_batch_stride + head_id * k_head_stride
    v_offset = batch_id * v_batch_stride + head_id * v_head_stride
    o_offset = batch_id * o_batch_stride + head_id * o_head_stride

    # Load query - for decoding, this is typically a single token
    # Shape: (BLOCK_D,)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_d < head_dim
    q_ptrs = Q + q_offset + offs_d * q_dim_stride
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Initialize accumulator for output
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    max_val = float("-inf")
    lse = 0.0

    # Iterate over blocks of KV
    for start_n in range(0, num_blocks * BLOCK_N, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < seqlen_kv

        # Load key block - shape: (BLOCK_N, BLOCK_D)
        k_ptrs = K + k_offset + n_offsets[:, None] * k_seq_stride + offs_d[None, :] * k_dim_stride
        k = tl.load(k_ptrs, mask=n_mask[:, None] & q_mask[None, :], other=0.0).to(tl.float32)

        # Compute Q @ K^T / sqrt(d)
        # q: (BLOCK_D,), k: (BLOCK_N, BLOCK_D)
        # q[None, :] * k: (BLOCK_N, BLOCK_D)
        scale_factor = 1.0 / tl.sqrt(head_dim.to(tl.float32))
        qk = tl.sum(q[None, :] * k, axis=1) * scale_factor

        # Apply causal mask (decoding is autoregressive)
        qk = tl.where(n_offsets <= start_n, qk, float("-inf"))

        # Softmax computation
        new_max = tl.maximum(max_val, tl.max(qk))
        exp_diff = tl.exp(qk - new_max)

        # Load value block - shape: (BLOCK_N, BLOCK_D)
        v_ptrs = V + v_offset + n_offsets[:, None] * v_seq_stride + offs_d[None, :] * v_dim_stride
        v = tl.load(v_ptrs, mask=n_mask[:, None] & q_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc = acc * exp(max - new_max) + sum(exp(qk - new_max) * V)
        acc = acc * tl.exp(max_val - new_max) + tl.sum(exp_diff[:, None] * v, axis=0)
        lse = lse * tl.exp(max_val - new_max) + tl.sum(exp_diff)
        max_val = new_max

    # Finalize: divide by sum of exponentials
    final_lse = max_val + tl.log(lse + 1e-8)

    # Store output
    offs_d = tl.arange(0, BLOCK_D)
    o_mask = offs_d < head_dim
    o_ptrs = O + o_offset + offs_d * o_dim_stride
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=o_mask)

    # Store log-sum-exp for backward pass
    softmax_lse_ptrs = softmax_lse + batch_id * num_heads + head_id
    tl.store(softmax_lse_ptrs, final_lse)


def flash_decoding(Q, K, V, scale=None):
    """
    FlashDecoding operator for efficient attention in LLM decoding.

    This is optimized for the decoding phase where we compute attention
    for a single query token against a large KV cache.

    Args:
        Q: Query tensor of shape (batch, num_heads, seq_q, head_dim)
        K: Key tensor of shape (batch, num_heads, seq_kv, head_dim)
        V: Value tensor of shape (batch, num_heads, seq_kv, head_dim)
        scale: Optional scale factor for attention scores

    Returns:
        Output tensor of shape (batch, num_heads, seq_q, head_dim)
    """
    logger.debug("GEMS FLASH_DECODING")

    batch_size = Q.shape[0]
    num_heads = Q.shape[1]
    seq_q = Q.shape[2]
    head_dim = Q.shape[3]
    seq_kv = K.shape[2]

    # For now, we only support seq_q = 1 (typical decoding case)
    # TODO: extend to support seq_q > 1
    if seq_q != 1:
        raise ValueError(f"FlashDecoding currently only supports seq_q=1, got seq_q={seq_q}")

    # Allocate output
    O = torch.empty_like(Q)
    softmax_lse = torch.empty((batch_size, num_heads, seq_q), device=Q.device, dtype=torch.float32)

    # Define block sizes
    BLOCK_M = 1  # For seq_q = 1
    BLOCK_N = min(triton.next_power_of_2(seq_kv), 1024)
    BLOCK_D = min(triton.next_power_of_2(head_dim), 128)

    # Compute number of blocks
    num_blocks = triton.cdiv(seq_kv, BLOCK_N)

    # Grid: (batch_size, num_heads)
    grid = (batch_size, num_heads)

    with torch_device_fn.device(Q.device):
        flash_decoding_kernel[grid](
            Q,
            K,
            V,
            O,
            softmax_lse,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            O.stride(3),
            seq_q,
            seq_kv,
            num_heads,
            head_dim,
            num_blocks,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

    return O