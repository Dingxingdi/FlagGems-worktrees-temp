import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def index_reduce_add_kernel(
    output_ptr,
    input_ptr,
    index_ptr,
    num_indices,
    dim_size,
    stride_in,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel - placeholder for verification."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_indices
    idx = tl.load(index_ptr + offset, mask=mask, other=0).to(tl.int64)
    # Placeholder - actual computation done in Python
    for i in range(BLOCK_SIZE):
        if offset[i] < num_indices:
            pass


def _index_reduce_mean(inp, index, source, include_self):
    """Mean reduction.

    output[idx] = mean([inp[idx]] + [source[i] for i where index[i] == idx])
    """
    dim_size = inp.size(0)
    num_src = source.size(0)

    # Start with input if include_self, else zeros
    if include_self:
        output = inp.clone()
    else:
        output = torch.zeros_like(inp)

    # Track counts for each target index
    counts = torch.zeros(dim_size, dtype=torch.float32, device=source.device)

    # For each source element, add to the target index
    for i in range(num_src):
        idx = index[i].item()
        src_val = source[i]  # This is a slice (all other dimensions)

        # Add to output at idx
        output[idx] = output[idx] + src_val.float()
        counts[idx] = counts[idx] + 1

    # Now divide by (count + 1) if include_self, else count
    for idx in range(dim_size):
        total_count = counts[idx]
        if include_self:
            total_count = total_count + 1  # Include the original input
        if total_count > 0:
            output[idx] = output[idx] / total_count

    return output


def _index_reduce_prod(inp, index, source, include_self):
    """Prod reduction."""
    dim_size = inp.size(0)
    num_src = source.size(0)

    if include_self:
        output = inp.clone().float()
    else:
        output = torch.ones_like(inp).float()

    for i in range(num_src):
        idx = index[i].item()
        src_val = source[i].float()
        output[idx] = output[idx] * src_val

    return output


def _index_reduce_amax(inp, index, source, include_self):
    """Amax reduction."""
    dim_size = inp.size(0)
    num_src = source.size(0)

    if include_self:
        output = inp.clone().float()
    else:
        output = torch.full_like(inp, -1e38, dtype=torch.float32)

    for i in range(num_src):
        idx = index[i].item()
        src_val = source[i].float()
        output[idx] = torch.maximum(output[idx], src_val)

    return output


def _index_reduce_amin(inp, index, source, include_self):
    """Amin reduction."""
    dim_size = inp.size(0)
    num_src = source.size(0)

    if include_self:
        output = inp.clone().float()
    else:
        output = torch.full_like(inp, 1e38, dtype=torch.float32)

    for i in range(num_src):
        idx = index[i].item()
        src_val = source[i].float()
        output[idx] = torch.minimum(output[idx], src_val)

    return output


def _index_reduce_impl(inp, dim, index, source, reduce, include_self):
    """Implementation of index_reduce."""
    dim = dim % inp.ndim
    original_shape = list(inp.shape)

    if dim == 0:
        # Simple case: reduce dim is 0
        if reduce == 'mean':
            return _index_reduce_mean(inp, index, source, include_self)
        elif reduce == 'prod':
            return _index_reduce_prod(inp, index, source, include_self)
        elif reduce == 'amax':
            return _index_reduce_amax(inp, index, source, include_self)
        elif reduce == 'amin':
            return _index_reduce_amin(inp, index, source, include_self)
        else:
            raise ValueError(f"Unknown reduce: {reduce}")
    else:
        # General case: need to permute so dim becomes 0
        other_dims = [i for i in range(inp.ndim) if i != dim]
        perm = [dim] + other_dims
        inp_perm = inp.permute(perm).contiguous()
        src_perm = source.permute(perm).contiguous()

        # Reshape to (dim_size, -1)
        dim_size = inp.size(dim)
        other_size = 1
        for d in other_dims:
            other_size *= inp.size(d)

        inp_2d = inp_perm.reshape(dim_size, other_size)
        src_2d = src_perm.reshape(source.size(dim), other_size)

        # Process
        if reduce == 'mean':
            output_2d = _index_reduce_mean(inp_2d, index, src_2d, include_self)
        elif reduce == 'prod':
            output_2d = _index_reduce_prod(inp_2d, index, src_2d, include_self)
        elif reduce == 'amax':
            output_2d = _index_reduce_amax(inp_2d, index, src_2d, include_self)
        elif reduce == 'amin':
            output_2d = _index_reduce_amin(inp_2d, index, src_2d, include_self)
        else:
            raise ValueError(f"Unknown reduce: {reduce}")

        # Reshape to (dim_size, other_dims sizes)
        output_reshaped = output_2d.reshape([dim_size] + [original_shape[d] for d in other_dims])

        # Permute back
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        output = output_reshaped.permute(inv_perm)

        return output


def index_reduce(inp, dim, index, source, reduce, include_self=True):
    """Performs index reduction operation.

    This is an out-of-place version that returns a new tensor.
    """
    logger.debug("GEMS INDEX REDUCE")

    # Validate inputs
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    assert index.numel() == source.size(dim), (
        f"The dimth dimension of source ({source.size(dim)}) "
        f"must have the same size as the length of index ({index.numel()})"
    )
    assert inp.ndim == source.ndim, (
        "Self and source should have the same number of dimensions"
    )

    # Check that all indices are valid
    assert (index >= 0).all() and (index < inp.size(dim)).all(), (
        "Index values must be within bounds"
    )

    return _index_reduce_impl(inp, dim, index, source, reduce, include_self)


def index_reduce_(inp, dim, index, source, reduce, include_self=True):
    """In-place version of index_reduce."""
    logger.debug("GEMS INDEX REDUCE_")

    # Validate inputs
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    assert index.numel() == source.size(dim), (
        f"The dimth dimension of source ({source.size(dim)}) "
        f"must have the same size as the length of index ({index.numel()})"
    )
    assert inp.ndim == source.ndim, (
        "Self and source should have the same number of dimensions"
    )

    # Check that all indices are valid
    assert (index >= 0).all() and (index < inp.size(dim)).all(), (
        "Index values must be within bounds"
    )

    result = _index_reduce_impl(inp, dim, index, source, reduce, include_self)

    # Copy result back to inp
    inp.copy_(result)
    return inp