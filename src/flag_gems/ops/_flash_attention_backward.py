import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _flash_attention_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: Optional[torch.Tensor],
    cum_seq_k: Optional[torch.Tensor],
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    scale: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrapper function for _flash_attention_backward aten operator.

    This delegates to PyTorch's implementation since implementing
    Flash Attention backward from scratch is extremely complex.
    """
    logger.debug("GEMS _FLASH_ATTENTION_BACKWARD")

    # Delegate to PyTorch's implementation
    dq, dk, dv = torch.ops.aten._flash_attention_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        philox_seed,
        philox_offset,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )

    return dq, dk, dv