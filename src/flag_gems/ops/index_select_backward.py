import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_select"))
@triton.jit
def index_select_backward_kernel(
    grad_output,
    index,
    output,
    M,
    N,
    index_len,
    stride_grad_out_0,
    stride_grad_out_1,
    stride_out_0,
    stride_out_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    index_select_backward kernel: scatter gradients back to original shape.

    Args:
        grad_output: gradient of the output from forward (shape: [..., index_len, ...])
        index: indices used in forward (shape: [index_len])
        output: output gradient tensor with shape = self_sizes
        M: product of dimensions except the indexed dimension
        N: size of the indexed dimension
        index_len: length of index
    """
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)

    # Row offset in the output (non-indexed dimensions)
    row_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = row_offsets < M

    # Column offsets in index dimension
    col_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < index_len

    # Load indices
    indices = tl.load(index + col_offsets, mask=col_mask, other=0)
    valid_lower_bound = indices >= 0
    valid_upper_bound = indices < N
    index_valid_mask = valid_lower_bound & valid_upper_bound

    # Load grad_output values
    grad_off = row_offsets * stride_grad_out_0 + col_offsets[None, :] * stride_grad_out_1
    grad_mask = row_mask and col_mask
    grad_values = tl.load(
        grad_output + grad_off, mask=grad_mask, other=0.0
    )

    # Compute output offsets - scatter to the indexed dimension
    out_off = row_offsets * stride_out_0 + indices[None, :] * stride_out_1
    final_mask = grad_mask & index_valid_mask

    # Atomic add to accumulate gradients
    tl.atomic_add(output + out_off, grad_values, mask=final_mask)


def index_select_backward(grad, self_sizes, dim, index):
    """
    Backward pass for index_select.

    Args:
        grad: gradient tensor from forward pass output
        self_sizes: sizes of the original input tensor (list or tuple)
        dim: dimension along which to index
        index: indices tensor

    Returns:
        Gradient tensor with shape = self_sizes
    """
    logger.debug("GEMS INDEX SELECT BACKWARD")

    # Handle negative dim
    dim = dim % len(self_sizes)
    N = self_sizes[dim]  # Size of indexed dimension in original input
    ndim = len(self_sizes)

    # Get index_len from grad tensor at the indexed dimension
    index_len = grad.shape[dim]

    # Compute M = product of all non-indexed dimensions
    M = 1
    for i, s in enumerate(grad.shape):
        if i != dim:
            M *= s

    # Use dim_compress to compress grad - this puts indexed dimension at the end
    grad_compressed = dim_compress(grad, dim)

    # For output:
    # After dim_compress, the order is: all dims except dim, then dim
    # So the output compressed shape should be: [dim 0, dim 1, ..., dim(dim-1), dim(dim+1), ..., dim(ndim-1), N]
    output_compressed_shape = []
    for i in range(ndim):
        if i != dim:
            output_compressed_shape.append(self_sizes[i])
    output_compressed_shape.append(N)

    # Create output tensor with compressed shape
    output_compressed = torch.zeros(
        output_compressed_shape, dtype=grad.dtype, device=grad.device
    )

    # Get strides from contiguous tensors
    stride_grad_out = grad_compressed.stride()
    stride_out = output_compressed.stride()

    # Launch kernel
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )

    index_select_backward_kernel[grid](
        grad_compressed,
        index,
        output_compressed,
        M,
        N,
        index_len,
        stride_grad_out[0],
        stride_grad_out[1],
        stride_out[0],
        stride_out[1],
    )

    # Now we need to permute output_compressed back to self_sizes
    # After dim_compress, order is: [0, 1, ..., dim-1, dim+1, ..., ndim-1, dim]
    # We need inverse: [0, 1, ..., dim-1, ndim-1, dim, dim+1, ..., ndim-2]
    inverse_perm = list(range(dim))  # 0 to dim-1
    inverse_perm.append(ndim - 1)  # put indexed dim (now at position ndim-1) back to original position
    for i in range(dim, ndim - 1):
        inverse_perm.append(i)

    # Permute and reshape
    output = output_compressed.permute(inverse_perm).reshape(self_sizes)

    return output