import logging

import torch

from flag_gems.ops.sum import sum as sum_impl
from flag_gems.ops.sum import sum_dim as sum_dim_impl
from flag_gems.ops.sum import sum_out as sum_out_impl

logger = logging.getLogger(__name__)


def ReduceSum(inp, dim=None, keepdim=False, *, dtype=None):
    """ReduceSum operator that sums elements of the tensor.

    This is equivalent to torch.sum but implemented as a separate operator
    for FlagGems.

    Args:
        inp: The input tensor.
        dim: The dimension(s) to reduce. If None, all dimensions are reduced.
        keepdim: Whether to keep the reduced dimensions.
        dtype: The desired data type of the output.

    Returns:
        The sum of elements.
    """
    logger.debug("GEMS ReduceSum")
    if dim is None:
        return sum_impl(inp, dtype=dtype)
    else:
        # Convert dim to list if it's an int
        dim_list = [dim] if isinstance(dim, int) else dim
        return sum_dim_impl(inp, dim=dim_list, keepdim=keepdim, dtype=dtype)


def ReduceSum_(inp, dim=None, keepdim=False, *, dtype=None):
    """In-place ReduceSum operator that sums elements of the tensor.

    Args:
        inp: The input tensor (will be modified in place).
        dim: The dimension(s) to reduce. If None, all dimensions are reduced.
        keepdim: Whether to keep the reduced dimensions.
        dtype: The desired data type of the output.

    Returns:
        The sum of elements.
    """
    logger.debug("GEMS ReduceSum_")
    if dim is None:
        out = torch.empty([], dtype=dtype or inp.dtype, device=inp.device)
        sum_out_impl(inp, dtype=dtype, out=out)
        return out
    else:
        # Convert dim to list if it's an int
        dim_list = [dim] if isinstance(dim, int) else dim
        return sum_dim_impl(inp, dim=dim_list, keepdim=keepdim, dtype=dtype)