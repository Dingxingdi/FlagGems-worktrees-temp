import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _scaled_dot_product_efficient_attention_forward(
    query,
    key,
    value,
    attn_bias,
    compute_log_sumexp,
    dropout_p,
    is_causal,
    scale,
):
    """Forward pass for _scaled_dot_product_efficient_attention.

    This implements the PyTorch 2.5+ efficient attention API:
    aten::_scaled_dot_product_efficient_attention(
        Tensor query, Tensor key, Tensor value, Tensor? attn_bias,
        bool compute_log_sumexp, float dropout_p=0., bool is_causal=False, *, float? scale=None
    ) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)

    Returns:
        Tuple of (output, log_sumexp, philox_seed, philox_offset)
    """
    # Defer imports to avoid circular import
    from flag_gems import runtime
    from flag_gems.ops.attention import _attn_fwd
    from flag_gems.runtime import torch_device_fn

    logger.debug("GEMS SCALED DOT PRODUCT EFFICIENT ATTENTION FORWARD")

    # Shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"

    # Compute scale
    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    # Get head dimensions
    q_head_num = query.shape[1]
    kv_head_num = key.shape[1]

    # Output tensor
    o = torch.empty_like(query, dtype=value.dtype)

    # log_sumexp tensor: shape (batch, num_heads, seq_len)
    log_sumexp = torch.empty(
        (query.shape[0], q_head_num, query.shape[2]),
        device=query.device,
        dtype=torch.float32,
    )

    # philox_seed and philox_offset: for dropout (currently unused)
    # When dropout_p=0, we return empty tensors
    if dropout_p > 0:
        # Generate random seed and offset for dropout
        philox_seed = torch.tensor([0], dtype=torch.long, device=query.device)
        philox_offset = torch.tensor([0], dtype=torch.long, device=query.device)
    else:
        philox_seed = torch.empty((1,), dtype=torch.long, device=query.device)
        philox_offset = torch.empty((1,), dtype=torch.long, device=query.device)

    stage = 3 if is_causal else 1

    # Process attention bias if provided
    if attn_bias is not None:
        HAS_ATTN_BIAS = True
        if attn_bias.dtype == torch.bool:
            attn_bias = attn_bias.to(query.dtype) * -1.0e6
        stride_attn_bias_batch = attn_bias.stride(0)
        stride_attn_bias_head = attn_bias.stride(1)
        stride_attn_bias_q_seqlen = attn_bias.stride(2)
        stride_attn_bias_kv_seqlen = attn_bias.stride(3)
    else:
        HAS_ATTN_BIAS = False
        stride_attn_bias_batch = 1
        stride_attn_bias_head = 1
        stride_attn_bias_q_seqlen = 1
        stride_attn_bias_kv_seqlen = 1

    grid = lambda args: (
        triton.cdiv(query.shape[2], args["BLOCK_M"]),
        query.shape[0] * query.shape[1],
        1,
    )

    # First get the output and max values using existing kernel
    M = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2]),
        device=query.device,
        dtype=torch.float32,
    )

    with torch_device_fn.device(query.device):
        _attn_fwd[grid](
            query,
            key,
            value,
            attn_bias,
            sm_scale,
            M,
            o,  #
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),  #
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),  #
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),  #
            stride_attn_bias_batch,
            stride_attn_bias_head,
            stride_attn_bias_q_seqlen,
            stride_attn_bias_kv_seqlen,  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            query.shape[0],
            q_head_num,
            kv_head_num,  #
            q_head_num // kv_head_num,  # group_head
            query.shape[2],  #
            key.shape[2],  #
            HEAD_DIM_K,  #
            STAGE=stage,  #
            HAS_ATTN_MASK=HAS_ATTN_BIAS,  #
        )

    # Compute log_sumexp from M (which contains max values)
    # M was computed as: m_i += log2(l_i), so log_sumexp = m_i
    log_sumexp = M

    return o, log_sumexp, philox_seed, philox_offset


def scaled_dot_product_efficient_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=False,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """Implements _scaled_dot_product_efficient_attention.

    This is a wrapper around the PyTorch efficient attention API.
    """
    o, log_sumexp, philox_seed, philox_offset = (
        _scaled_dot_product_efficient_attention_forward(
            query,
            key,
            value,
            attn_bias,
            compute_log_sumexp,
            dropout_p,
            is_causal,
            scale,
        )
    )
    return o, log_sumexp, philox_seed, philox_offset