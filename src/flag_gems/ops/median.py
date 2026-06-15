import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import topk
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def median(inp, dim=None, keepdim=False):
    """Compute the median of the input tensor.

    Args:
        inp: Input tensor
        dim: Dimension to compute median along. If None, computes global median.
        keepdim: Whether to keep the reduced dimension.

    Returns:
        If dim is None: scalar tensor with the median value
        If dim is not None: namedtuple (values, indices) where values contains
            the median and indices contains the indices in the original tensor.
    """
    logger.debug("GEMS MEDIAN")

    if dim is None:
        # Global median - flatten the tensor and find median
        M = inp.numel()
        if M == 0:
            # Empty tensor
            return torch.tensor(float('nan'), dtype=inp.dtype, device=inp.device)

        # For median, we need k = M // 2 + 1 smallest elements
        # The median is at index (M - 1) // 2 (lower median for even M)
        k = M // 2 + 1
        median_idx = (M - 1) // 2

        # Flatten input
        inp_flat = inp.flatten()

        # Use topk with largest=False to get k smallest elements
        # Then get the element at median_idx
        values, indices = topk(inp_flat, k=k, dim=0, largest=False, sorted=True)
        result = values[median_idx]

        # Preserve the original shape if keepdim was conceptually applied
        # But for global median, PyTorch returns a scalar
        return result
    else:
        # Dimension-based median
        # Handle negative dim
        dim = dim % inp.ndim

        N = inp.shape[dim]
        if N == 0:
            # Empty dimension
            out_shape = list(inp.shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                out_shape.pop(dim)
            values = torch.full(out_shape, float('nan'), dtype=inp.dtype, device=inp.device)
            indices = torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
            return values, indices

        # For median, we need k = N // 2 + 1 smallest elements along the dimension
        # The median is at index (N - 1) // 2 (lower median for even N)
        k = N // 2 + 1
        median_idx = (N - 1) // 2

        # Use topk to get k smallest elements along the specified dimension
        # topk currently only supports last dimension, so we need to transpose
        if dim != inp.ndim - 1:
            # Move the reduction dimension to the last axis
            perm = list(range(inp.ndim))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            inp_perm = inp.permute(perm)
            values_perm, indices_perm = topk(inp_perm, k=k, dim=-1, largest=False, sorted=True)

            # Get the median element
            values_perm = values_perm[..., median_idx:median_idx+1]
            indices_perm = indices_perm[..., median_idx:median_idx+1]

            # Transpose back
            values = values_perm.permute(perm)
            indices = indices_perm.permute(perm)
        else:
            values, indices = topk(inp, k=k, dim=dim, largest=False, sorted=True)
            values = values[..., median_idx:median_idx+1]
            indices = indices[..., median_idx:median_idx+1]

        if not keepdim:
            values = values.squeeze(dim=dim)
            indices = indices.squeeze(dim=dim)

        return values, indices