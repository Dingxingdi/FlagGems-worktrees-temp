import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def linalg_vander_kernel(
    x_ptr,
    output_ptr,
    n,
    N,
    batch_size,
    x_stride_batch,
    x_stride_n,
    out_stride_batch,
    out_stride_n,
    out_stride_N,
):
    """
    Generate Vandermonde matrix.
    For input x of shape (batch, n), output is (batch, n, N) where N is the number of columns.
    output[i, j, k] = x[i, j]^k for k in [0, N-1]
    """
    batch_idx = tle.program_id(0)
    n_idx = tle.program_id(1)

    if batch_idx >= batch_size or n_idx >= n:
        return

    # Load input value x[batch_idx, n_idx]
    x_offset = batch_idx * x_stride_batch + n_idx * x_stride_n
    x_val = tl.load(x_ptr + x_offset)

    # Compute output values: [1, x, x^2, ..., x^(N-1)]
    # First column is always 1.0 (in float32, will be converted later)
    out_offset_0 = batch_idx * out_stride_batch + n_idx * out_stride_n
    tl.store(output_ptr + out_offset_0, tl.full((), 1.0, tl.float32))

    # Compute remaining columns
    if N > 1:
        # Start with x^1 = x
        curr_power = x_val
        for col_idx in range(1, N):
            out_offset = batch_idx * out_stride_batch + n_idx * out_stride_n + col_idx * out_stride_N
            tl.store(output_ptr + out_offset, curr_power)
            # Update power: x^(col_idx+1) = x^col_idx * x
            curr_power = curr_power * x_val


def linalg_vander(x: torch.Tensor, N: int = None) -> torch.Tensor:
    logger.debug("GEMS LINALG_VANDER")

    # Handle N parameter
    if N is None:
        N = x.shape[-1]

    # Get input shape info
    n = x.shape[-1]  # Last dimension is the vector length

    # Remember if input was 1D
    is_1d_input = x.ndim == 1

    # Handle batch dimensions
    if x.ndim == 1:
        # 1D input: (n,) -> output (n, N)
        batch_size = 1
        x = x.unsqueeze(0)  # Make it (1, n)
    else:
        # Multi-dim input: (*, n) -> output (*, n, N)
        batch_size = x.numel() // n
        x = x.reshape(batch_size, n)

    x = x.contiguous()

    # Convert to float32 for computation to avoid precision issues with fp16/bf16
    input_dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)

    # Create output tensor in float32
    output = torch.empty((batch_size, n, N), dtype=torch.float32, device=x.device)

    # Define grid
    # Grid: (batch_size, n)
    grid = (batch_size, n)

    with torch_device_fn.device(x.device):
        linalg_vander_kernel[grid](
            x,
            output,
            n,
            N,
            batch_size,
            x.stride(0),
            x.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )

    # Convert output back to original dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        output = output.to(input_dtype)

    # Reshape to original batch shape + (n, N)
    if is_1d_input:
        # Original was 1D, return (n, N)
        return output.squeeze(0)
    else:
        # Original was multi-dim, restore batch shape
        original_batch_shape = x.shape[:-1]
        return output.view(*original_batch_shape, n, N)