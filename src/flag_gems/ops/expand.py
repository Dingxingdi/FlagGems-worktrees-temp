import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.shape_utils import broadcast_shapes, broadcasted_stride
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def expand_func(x):
    return x


def expand(input: torch.Tensor, size) -> torch.Tensor:
    logger.debug("GEMS EXPAND")

    if not isinstance(size, (list, tuple, torch.Size)):
        raise TypeError("expand size must be a list/tuple/torch.Size of ints")

    size = list(size)
    in_shape = list(input.shape)
    in_strides = list(input.stride())

    out_ndim = len(size)
    in_ndim = len(in_shape)

    if in_ndim > out_ndim:
        raise RuntimeError(
            f"expand: requested size has fewer dimensions ({out_ndim}) than input ({in_ndim})"
        )

    # Pad input shape/strides on the left to match output ndim
    if in_ndim < out_ndim:
        pad = out_ndim - in_ndim
        in_shape = [1] * pad + in_shape
        # For padded (new) leading dims, stride effectively is 0 since they will be broadcast
        in_strides = [0] * pad + in_strides

    # Resolve -1 and validate broadcastability
    out_shape = []
    for d in range(out_ndim):
        req = size[d]
        src = in_shape[d]
        if req == -1:
            target = src
        else:
            target = req
        if src != target and src != 1:
            raise RuntimeError(
                f"The expanded size of the tensor ({target}) must match the existing size ({src}) at non-singleton "
                f"dimension {d}. Target sizes must be the same, or -1, or the size of dimension in the original tensor must be 1."
            )
        out_shape.append(int(target))

    # Compute broadcasted strides for the expanded view
    expanded_strides = broadcasted_stride(tuple(in_shape), tuple(in_strides), tuple(out_shape))

    # Create StridedBuffer to represent the expanded view without calling torch.expand
    expanded_input = StridedBuffer(input, tuple(out_shape), expanded_strides)

    # Allocate output tensor
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    # Use the pointwise_dynamic function to copy data
    # The expand_func will read from the expanded_input with proper broadcasting
    overload = expand_func.instantiate(out.ndim)
    overload(expanded_input, out0=out)

    return out