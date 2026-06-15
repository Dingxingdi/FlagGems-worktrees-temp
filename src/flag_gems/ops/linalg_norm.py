import logging
import torch

from flag_gems.ops.vector_norm import vector_norm

logger = logging.getLogger(__name__)


def linalg_norm(A, ord=None, dim=None, keepdim=False, *, dtype=None, out=None):
    """
    Computes a vector or matrix norm.

    This is a wrapper that dispatches to vector_norm for common vector norm cases.
    For matrix norms or unsupported cases, falls back to PyTorch implementation.
    """
    logger.debug("GEMS LINALG_NORM")

    # Handle dtype
    if dtype is not None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        elif not isinstance(dtype, torch.dtype):
            dtype = torch.float32
    else:
        dtype = A.dtype

    # For unsupported dtypes, use PyTorch directly
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        result = torch.linalg.norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
        if out is not None:
            return out
        return result

    # Check if this is a case we can handle with vector_norm
    # vector_norm can handle:
    # - dim is None (flatten)
    # - dim is an integer
    # - dim is a single-element list/tuple

    can_use_vector_norm = True

    # Check for matrix norm cases that we should fallback on
    if dim is None:
        # Flatten case - use vector_norm
        pass
    elif isinstance(dim, int):
        # Single dimension - use vector_norm
        pass
    elif isinstance(dim, (list, tuple)):
        if len(dim) == 1:
            # Single dimension in list form - use vector_norm
            pass
        elif len(dim) == 2:
            # Matrix norm - fallback to PyTorch
            can_use_vector_norm = False
        else:
            # Multi-dim reduction that's not a matrix norm - use PyTorch
            can_use_vector_norm = False
    else:
        can_use_vector_norm = False

    # Check ord values that vector_norm supports
    if can_use_vector_norm and ord is not None:
        if ord not in [2, 1, 0, float('inf'), -float('inf')] and not isinstance(ord, (int, float)):
            # Other ord values (like 'fro', 'nuc') - fallback to PyTorch
            can_use_vector_norm = False
        # For string ord values
        if isinstance(ord, str):
            can_use_vector_norm = False

    if not can_use_vector_norm:
        # Fallback to PyTorch - use torch.ops.aten to bypass FlagGems dispatch
        try:
            with torch.no_grad():
                result = torch.ops.aten.linalg_norm(A, ord, dim, keepdim, dtype=dtype)
        except Exception:
            # Direct aten call failed, try regular torch call
            with torch.no_grad():
                result = torch.linalg.norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
        if out is not None:
            out.copy_(result) if result.numel() == out.numel() else out.resize_(result.shape).copy_(result)
            return out
        return result

    # For vector norms, use vector_norm
    try:
        # Normalize dim for vector_norm - it needs dim to be None or a list
        norm_dim = dim
        if dim is None:
            # Flatten case - use all dimensions
            norm_dim = list(range(A.ndim))
        elif isinstance(dim, int):
            # Convert integer dim to single-element list
            norm_dim = [dim]
        elif isinstance(dim, (list, tuple)) and len(dim) == 1:
            # Single element list is fine
            pass

        result = vector_norm(A, ord=ord if ord is not None else 2, dim=norm_dim, keepdim=keepdim, dtype=dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result
    except Exception as e:
        logger.debug(f"vector_norm failed, falling back to PyTorch: {e}")
        with torch.no_grad():
            result = torch.linalg.norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
        if out is not None:
            return out
        return result


def linalg_norm_out(A, ord=None, dim=None, keepdim=False, *, dtype=None, out):
    """Out-place version of linalg_norm."""
    logger.debug("GEMS LINALG_NORM_OUT")
    return linalg_norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype, out=out)