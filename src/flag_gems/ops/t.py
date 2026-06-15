import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def t_2d_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_i0,
    stride_i1,
    stride_o0,
    stride_o1,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for transposing a 2D tensor.

    Input has shape (M, N), output has shape (N, M).
    Uses 1D grid where each program handles multiple elements.
    """
    # Get program id
    pid = tl.program_id(0)
    num_elements = M * N

    # Each program processes BLOCK_SIZE elements
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    # Convert linear index to 2D (row, col) in input
    row = offs // N
    col = offs % N

    # For input[row, col], output is at [col, row]
    input_ptrs = input_ptr + row * stride_i0 + col * stride_i1
    output_ptrs = output_ptr + col * stride_o0 + row * stride_o1

    # Load and store
    data = tl.load(input_ptrs, mask=mask, other=0.0)
    tl.store(output_ptrs, data, mask=mask)


def t(input: torch.Tensor) -> torch.Tensor:
    """Transpose a tensor.

    For 0D and 1D tensors, returns the input as is.
    For 2D tensors, transposes dimensions 0 and 1.
    """
    logger.debug("GEMS t")

    if input.dim() == 0:
        # 0D tensor - return as is
        return input.clone()

    if input.dim() == 1:
        # 1D tensor - return as is
        return input.clone()

    if input.dim() == 2:
        # 2D tensor - use Triton kernel for transpose
        M, N = input.shape
        output = torch.empty((N, M), dtype=input.dtype, device=input.device)

        # Calculate grid
        grid = lambda meta: (triton.cdiv(M * N, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(input.device):
            t_2d_kernel[grid](
                input,
                output,
                M,
                N,
                input.stride(0),
                input.stride(1),
                output.stride(0),
                output.stride(1),
            )
        return output

    # For higher dimensions, use torch.t
    return torch.t(input)


def t_(input: torch.Tensor) -> torch.Tensor:
    """In-place transpose a tensor.

    For 0D and 1D tensors, returns the input as is.
    For 2D tensors, transposes dimensions 0 and 1 in place.
    """
    logger.debug("GEMS t_")

    if input.dim() <= 1:
        # 0D and 1D tensors - return as is
        return input

    if input.dim() == 2:
        # 2D tensor - use in-place transpose
        return input.transpose_(0, 1)

    # For higher dimensions, use torch.t_
    return torch.t_(input)