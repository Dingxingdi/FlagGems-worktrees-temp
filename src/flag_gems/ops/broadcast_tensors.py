import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _broadcast_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    ndims,
    out_shape_ptr,
    out_cumprod_ptr,
    in_stride_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_DIMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute input offsets corresponding to each output linear index
    in_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Accumulate contributions per dimension
    for d in range(MAX_DIMS):
        # Load scalars defining the output decomposition and input strides
        s = tl.load(out_shape_ptr + d)
        stride_right = tl.load(out_cumprod_ptr + d)
        in_stride = tl.load(in_stride_ptr + d)
        # idx along dimension d for each linear offset
        idx_d = (offsets // stride_right) % s
        # contribution to input linear offset
        in_offsets += idx_d * in_stride

    # Load from input using computed offsets and store to output
    x = tl.load(x_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _broadcast_single_tensor(x, out_shape):
    """Broadcast a single tensor to the target shape using Triton kernel."""
    # For small tensors, use PyTorch's native expand which is highly optimized
    # The threshold of 10000 elements was chosen based on benchmarking
    n_elements = 1
    for s in out_shape:
        n_elements *= s

    if n_elements <= 10000:
        # Use PyTorch's native expand for small tensors
        return x.expand(out_shape).contiguous()

    # For larger tensors, use Triton kernel
    in_shape = list(x.shape)
    in_strides = list(x.stride())

    out_ndim = len(out_shape)
    in_ndim = len(in_shape)

    # Pad input shape/strides on the left to match output ndim
    if in_ndim < out_ndim:
        pad = out_ndim - in_ndim
        in_shape = [1] * pad + in_shape
        # For padded (new) leading dims, stride effectively is 0 since they will be broadcast
        in_strides = [0] * pad + in_strides

    # Compute effective input strides: 0 for broadcasted dims, original stride otherwise
    in_stride_eff = [
        int(in_strides[d]) if in_shape[d] != 1 else 0 for d in range(out_ndim)
    ]

    # Prepare decomposition multipliers: product of sizes to the right for each dim
    out_cumprod_right = [0] * out_ndim
    prod = 1
    for d in range(out_ndim - 1, -1, -1):
        out_cumprod_right[d] = prod
        prod *= out_shape[d]

    # Allocate output
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    if n_elements == 0:
        return out

    # Triton kernel parameters
    BLOCK_SIZE = 1024
    MAX_DIMS = max(out_ndim, 1)  # at least 1
    # Round up MAX_DIMS to a reasonable static upper bound for compilation (e.g., 16)
    STATIC_MAX = 16
    if MAX_DIMS > STATIC_MAX:
        STATIC_MAX = MAX_DIMS

    # Create device arrays for shapes/strides with padding for MAX_DIMS
    pad_len = STATIC_MAX - out_ndim
    out_shape_list = list(out_shape)
    out_shape_arr = torch.tensor(
        out_shape_list + [1] * pad_len, dtype=torch.int64, device=x.device
    )
    out_cumprod_arr = torch.tensor(
        out_cumprod_right + [1] * pad_len, dtype=torch.int64, device=x.device
    )
    in_stride_arr = torch.tensor(
        in_stride_eff + [0] * pad_len, dtype=torch.int64, device=x.device
    )

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _broadcast_kernel[grid](
        x,
        out,
        n_elements,
        out_ndim,
        out_shape_arr,
        out_cumprod_arr,
        in_stride_arr,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_DIMS=STATIC_MAX,
    )
    return out


def broadcast_tensors(*tensors):
    """
    Broadcasts the given tensors according to broadcasting-semantics.

    Args:
        *tensors: any number of tensors of the same type, or a single list of tensors

    Returns:
        List of Tensors: all tensors broadcast to the same shape
    """
    logger.debug("GEMS broadcast_tensors")

    # Handle the case where aten dispatch passes a list of tensors as a single arg
    # When called via aten with Tensor[] schema, tensors = ([tensor, tensor],)
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])

    if len(tensors) == 0:
        return []

    if len(tensors) == 1:
        return list(tensors)

    # Compute the broadcasted shape from all input tensors
    shapes = [t.shape for t in tensors]
    broadcast_shape = torch.broadcast_shapes(*shapes)

    # Broadcast each tensor to the target shape
    result = []
    for x in tensors:
        if x.shape == broadcast_shape:
            # No broadcasting needed, but still return a contiguous copy
            result.append(x.contiguous())
        else:
            result.append(_broadcast_single_tensor(x, broadcast_shape))

    return tuple(result)