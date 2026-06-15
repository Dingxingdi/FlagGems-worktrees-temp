import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def fused_rope_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    position_ids_ptr,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    cos_stride_s,
    sin_stride_s,
    p_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
    USE_POSITION_IDS: tl.constexpr,
):
    s_id = tle.program_id(0)

    if USE_POSITION_IDS:
        pos_ptr = position_ids_ptr + s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    else:
        pos_id = s_id % seq_len

    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM

    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    # Process query
    oq_ptr += s_id * oq_stride_s
    q_ptr += s_id * q_stride_s

    for off_h in range(0, NUM_Q_HEADS):
        ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)
        output_offs = off_h * oq_stride_h + (ordered_block * oq_stride_d)

        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        y = q * cos + rotated_q * sin
        tl.store(oq_ptr + output_offs, y, mask=mask)

    # Process key
    ok_ptr += s_id * ok_stride_s
    k_ptr += s_id * k_stride_s

    for off_h in range(0, NUM_K_HEADS):
        ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)
        output_offs = off_h * ok_stride_h + (ordered_block * ok_stride_d)

        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        y = k * cos + rotated_k * sin
        tl.store(ok_ptr + output_offs, y, mask=mask)


@libentry()
@triton.jit
def fused_rope_inplace_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    position_ids_ptr,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    cos_stride_s,
    sin_stride_s,
    p_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
    USE_POSITION_IDS: tl.constexpr,
):
    s_id = tle.program_id(0)

    if USE_POSITION_IDS:
        pos_ptr = position_ids_ptr + s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    else:
        pos_id = s_id % seq_len

    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM

    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    # Process query in-place
    q_ptr += s_id * q_stride_s

    for off_h in range(0, NUM_Q_HEADS):
        ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)

        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        y = q * cos + rotated_q * sin
        tl.store(q_ptr + ordered_cols, y, mask=mask)

    # Process key in-place
    k_ptr += s_id * k_stride_s

    for off_h in range(0, NUM_K_HEADS):
        ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)

        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        y = k * cos + rotated_k * sin
        tl.store(k_ptr + ordered_cols, y, mask=mask)


def fused_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply fused rotary position embedding to query and key tensors.

    Args:
        query: Query tensor with shape (..., q_heads, head_dim)
        key: Key tensor with shape (..., k_heads, head_dim)
        cos: Cosine cache with shape (max_seq_len, head_dim // 2)
        sin: Sine cache with shape (max_seq_len, head_dim // 2)
        position_ids: Optional position ids with shape (...,)
        rotary_interleaved: Whether to use interleaved rotary layout

    Returns:
        Tuple of (query_output, key_output) with the same shape as input
    """
    logger.debug("GEMS Fused_RoPE")

    assert query.dim() >= 3, f"query must have at least 3 dims, got {query.dim()}"
    assert key.dim() >= 3, f"key must have at least 3 dims, got {key.dim()}"

    assert (
        key.shape[-1] == query.shape[-1]
    ), f"query and key must have the same last dimension, got {query.shape} and {key.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == query.shape[-1]
    ), f"cos/sin dim must be half of query/key dim, got {cos.shape} and {query.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    q_shape = query.shape
    k_shape = key.shape

    assert (
        query.shape[:-2] == key.shape[:-2]
    ), f"query and key must have the same leading dimensions, got {query.shape[:-2]} and {key.shape[:-2]}"

    if position_ids is None:
        seq_len = query.shape[-3]
    else:
        assert position_ids.shape == query.shape[:-2], (
            f"position_ids must have the same leading dimensions as query, got {position_ids.shape} and {query.shape[:-2]}"
        )
        position_ids = position_ids.view(-1)
        seq_len = None

    query = query.view(-1, query.shape[-2], query.shape[-1])
    key = key.view(-1, key.shape[-2], key.shape[-1])

    n_tokens, q_heads, head_dim = query.shape
    _, k_heads, _ = key.shape

    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    query_embed = torch.empty_like(query)
    key_embed = torch.empty_like(key)

    grid = (n_tokens,)

    fused_rope_kernel[grid](
        query_embed,
        key_embed,
        query,
        key,
        cos,
        sin,
        position_ids if position_ids is not None else cos.new_zeros(1),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        query_embed.stride(0),
        query_embed.stride(1),
        query_embed.stride(2),
        key_embed.stride(0),
        key_embed.stride(1),
        key_embed.stride(2),
        cos.stride(0),
        sin.stride(0),
        position_ids.stride(0) if position_ids is not None else 0,
        seq_len,
        q_heads,
        k_heads,
        head_dim,
        padded_head_dim,
        rotary_interleaved,
        MAX_POSITION_EMBEDDINGS=cos.shape[0],
        USE_POSITION_IDS=position_ids is not None,
    )

    query_embed = query_embed.view(q_shape)
    key_embed = key_embed.view(k_shape)
    return query_embed, key_embed


def fused_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply fused rotary position embedding to query and key tensors in-place.

    Args:
        query: Query tensor with shape (..., q_heads, head_dim) to be modified in-place
        key: Key tensor with shape (..., k_heads, head_dim) to be modified in-place
        cos: Cosine cache with shape (max_seq_len, head_dim // 2)
        sin: Sine cache with shape (max_seq_len, head_dim // 2)
        position_ids: Optional position ids with shape (...,)
        rotary_interleaved: Whether to use interleaved rotary layout

    Returns:
        Tuple of (query_output, key_output) with the same shape as input
    """
    logger.debug("GEMS Fused_RoPE_")

    assert query.dim() >= 3, f"query must have at least 3 dims, got {query.dim()}"
    assert key.dim() >= 3, f"key must have at least 3 dims, got {key.dim()}"

    assert (
        key.shape[-1] == query.shape[-1]
    ), f"query and key must have the same last dimension, got {query.shape} and {key.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == query.shape[-1]
    ), f"cos/sin dim must be half of query/key dim, got {cos.shape} and {query.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    q_shape = query.shape
    k_shape = key.shape

    assert (
        query.shape[:-2] == key.shape[:-2]
    ), f"query and key must have the same leading dimensions, got {query.shape[:-2]} and {key.shape[:-2]}"

    if position_ids is None:
        seq_len = query.shape[-3]
    else:
        assert position_ids.shape == query.shape[:-2], (
            f"position_ids must have the same leading dimensions as query, got {position_ids.shape} and {query.shape[:-2]}"
        )
        position_ids = position_ids.view(-1)
        seq_len = None

    query = query.view(-1, query.shape[-2], query.shape[-1])
    key = key.view(-1, key.shape[-2], key.shape[-1])

    n_tokens, q_heads, head_dim = query.shape
    _, k_heads, _ = key.shape

    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    grid = (n_tokens,)

    fused_rope_inplace_kernel[grid](
        query,
        key,
        cos,
        sin,
        position_ids if position_ids is not None else cos.new_zeros(1),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        cos.stride(0),
        sin.stride(0),
        position_ids.stride(0) if position_ids is not None else 0,
        seq_len,
        q_heads,
        k_heads,
        head_dim,
        padded_head_dim,
        rotary_interleaved,
        MAX_POSITION_EMBEDDINGS=cos.shape[0],
        USE_POSITION_IDS=position_ids is not None,
    )

    query = query.view(q_shape)
    key = key.view(k_shape)
    return query, key