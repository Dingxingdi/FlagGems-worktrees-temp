import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _mha_fwd_kernel(
    Q,
    K,
    V,
    Out,
    M,  # softmax scores
    stride_q_batch,
    stride_q_head,
    stride_q_seqlen,
    stride_q_headsize,
    stride_k_batch,
    stride_k_head,
    stride_k_seqlen,
    stride_k_headsize,
    stride_v_batch,
    stride_v_head,
    stride_v_seqlen,
    stride_v_headsize,
    stride_o_batch,
    stride_o_head,
    stride_o_seqlen,
    stride_o_headsize,
    Z,  # batch size
    H,  # num heads
    Q_CTX,  # query sequence length
    KV_CTX,  # key/value sequence length
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Multi-Head Attention forward kernel.

    Input shapes:
        Q: (batch, num_heads, seq_len_q, head_dim)
        K: (batch, num_heads, seq_len_k, head_dim)
        V: (batch, num_heads, seq_len_k, head_dim)
    Output shape:
        Out: (batch, num_heads, seq_len_q, head_dim)
    """
    # Get program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // H
    head_id = off_hz % H

    # Compute offsets
    q_offset = batch_id * stride_q_batch + head_id * stride_q_head
    kv_offset = batch_id * stride_k_batch + head_id * stride_k_head
    o_offset = batch_id * stride_o_batch + head_id * stride_o_head

    offs_headsize = tl.arange(0, HEAD_DIM)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Load query
    q_load_mask = offs_m < Q_CTX
    Q_block_ptr = (
        Q
        + q_offset
        + offs_m[:, None] * stride_q_seqlen
        + offs_headsize[None, :] * stride_q_headsize
    )
    query = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0)

    # Initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Scale factor
    qk_scale = 1.0 / (HEAD_DIM**0.5)
    LOG2E = 1.44269504  # log2(e) constant

    # Load K and V block pointers
    K_block_ptr = (
        K
        + kv_offset
        + offs_n[None, :] * stride_k_seqlen
        + offs_headsize[:, None] * stride_k_headsize
    )
    V_block_ptr = (
        V
        + kv_offset
        + offs_n[:, None] * stride_v_seqlen
        + offs_headsize[None, :] * stride_v_headsize
    )

    # Loop over key/value
    hi = KV_CTX
    for start_n in range(0, hi, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX

        # Load K
        key = tl.load(K_block_ptr, mask=kv_load_mask[None, :], other=0.0)

        # Compute QK^T
        qk = tl.dot(query, key, allow_tf32=False)
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))

        # Apply scaling
        qk = qk * qk_scale * LOG2E

        # Compute softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # Update accumulator
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V and compute attention
        value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        p = p.to(value.dtype)
        acc = tl.dot(p, value, acc, allow_tf32=False)

        # Update m_i
        m_i = m_ij

        # Increment pointers
        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen

    # Normalize
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # Store output
    O_block_ptr = (
        Out
        + o_offset
        + offs_m[:, None] * stride_o_seqlen
        + offs_headsize[None, :] * stride_o_headsize
    )
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), mask=q_load_mask[:, None])

    # Store max scores
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=q_load_mask)


def multi_head_attention_mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
):
    """
    Multi-Head Attention (MHA) implementation.

    This is a custom implementation that follows the attention mechanism:

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len_k, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len_k, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (currently not supported)
        is_causal: Whether to use causal masking
        scale: Optional scale factor (defaults to 1/sqrt(head_dim))

    Returns:
        output: Attention output of shape (batch, num_heads, seq_len_q, head_dim)
    """
    logger.debug("GEMS MULTI_HEAD_ATTENTION_MHA")

    # Validate shapes
    assert query.dim() == 4, f"query must be 4D, got {query.dim()}D"
    assert key.dim() == 4, f"key must be 4D, got {key.dim()}D"
    assert value.dim() == 4, f"value must be 4D, got {value.dim()}D"

    batch, num_heads, seq_len_q, head_dim = query.shape
    _, _, seq_len_k, _ = key.shape
    _, _, _, head_dim_v = value.shape

    assert head_dim == head_dim_v, f"head_dim mismatch: {head_dim} vs {head_dim_v}"
    assert dropout_p == 0.0, "dropout_p > 0.0 is not supported yet"

    # Create output tensor
    output = torch.empty_like(query)

    # Create tensor for max scores (needed for backward pass)
    M = torch.empty(
        (batch, num_heads, seq_len_q),
        device=query.device,
        dtype=torch.float32,
    )

    # Set scale
    if scale is None:
        sm_scale = 1.0 / (head_dim**0.5)
    else:
        sm_scale = scale

    # Grid configuration
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (
        triton.cdiv(seq_len_q, BLOCK_M),
        batch * num_heads,
    )

    # Launch kernel
    _mha_fwd_kernel[grid](
        query,
        key,
        value,
        output,
        M,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        batch,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        BLOCK_M,
        BLOCK_N,
    )

    return output