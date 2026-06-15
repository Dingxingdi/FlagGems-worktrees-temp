import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 512}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 1024}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_D": 2048}, num_stages=3, num_warps=8),
    ],
    key=["d"],
)
@triton.jit
def _cdist_forward_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    n,
    m,
    d,
    p,
    stride_x1_b,
    stride_x1_n,
    stride_x1_d,
    stride_x2_b,
    stride_x2_m,
    stride_x2_d,
    stride_out_b,
    stride_out_n,
    stride_out_m,
    BLOCK_D: tl.constexpr,
):
    """
    Compute p-norm distances between rows of x1 and x2.

    x1: (batch, n, d) or (n, d)
    x2: (batch, m, d) or (m, d)
    out: (batch, n, m) or (n, m)

    Each program computes result[i, j] = ||x1[i,:] - x2[j,:]||_p
    """
    batch_pid = tle.program_id(2)

    # Get position of this program - each program computes one (i, j) pair
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    # Check if this program is within bounds
    if pid_m >= n or pid_n >= m:
        return

    # Compute the output offset
    out_offset = (
        batch_pid * stride_out_b
        + pid_m * stride_out_n
        + pid_n * stride_out_m
    )

    # Load x1 row: x1[batch, pid_m, :]
    x1_base = (
        batch_pid * stride_x1_b
        + pid_m * stride_x1_n
    )
    # Load x2 row: x2[batch, pid_n, :]
    x2_base = (
        batch_pid * stride_x2_b
        + pid_n * stride_x2_m
    )

    # Compute p-norm distance by accumulating along d
    acc = 0.0

    # Process along d dimension
    for d_start in range(0, d, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_offsets < d

        # Load x1 row elements
        x1_ptrs = x1_ptr + x1_base + d_offsets * stride_x1_d
        x1_vals = tl.load(x1_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        # Load x2 row elements
        x2_ptrs = x2_ptr + x2_base + d_offsets * stride_x2_d
        x2_vals = tl.load(x2_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        # Compute difference and accumulate |x1 - x2|^p
        diff = x1_vals - x2_vals
        abs_diff = tl.abs(diff)

        if p == 2.0:
            acc += tl.sum(diff * diff)
        elif p == 1.0:
            acc += tl.sum(abs_diff)
        else:
            # x^p = exp(p * log(x)) for x > 0
            acc += tl.sum(tl.exp(p * tl.log(abs_diff + 1e-9)))

    # Apply final transformation: result = acc^(1/p)
    if p == 2.0:
        result = tl.sqrt(acc)
    elif p == 1.0:
        result = acc
    else:
        # x^(1/p) = exp((1/p) * log(x))
        result = tl.exp((1.0 / p) * tl.log(acc + 1e-9))

    # Store result
    tl.store(out_ptr + out_offset, result)


def _cdist_forward(x1, x2, p=2.0, compute_mode=None):
    logger.debug("GEMS CDIST_FORWARD")

    # Validate input dimensions
    x1_shape = x1.shape
    x2_shape = x2.shape

    assert x1.dim() in (2, 3), f"x1 must be 2D or 3D, got {x1.dim()}D"
    assert x2.dim() in (2, 3), f"x2 must be 2D or 3D, got {x2.dim()}D"
    assert x1.dim() == x2.dim(), f"x1 and x2 must have same dim, got {x1.dim()} and {x2.dim()}"
    assert x1_shape[-1] == x2_shape[-1], f"Feature dimension mismatch: {x1_shape[-1]} vs {x2_shape[-1]}"

    # Determine batch dimension
    has_batch = x1.dim() == 3
    if has_batch:
        batch_size = x1_shape[0]
        assert x2_shape[0] == batch_size, f"Batch size mismatch: {batch_size} vs {x2_shape[0]}"
        n, d = x1_shape[1], x1_shape[2]
        m = x2_shape[1]
    else:
        batch_size = 1
        n, d = x1_shape[0], x1_shape[1]
        m = x2_shape[0]

    # Handle empty tensors
    if n == 0 or m == 0 or d == 0:
        out_shape = (batch_size, n, m) if has_batch else (n, m)
        return torch.empty(out_shape, dtype=x1.dtype, device=x1.device)

    # Allocate output
    out_shape = (batch_size, n, m) if has_batch else (n, m)
    out = torch.empty(out_shape, dtype=x1.dtype, device=x1.device)

    # Make contiguous
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    # Launch kernel - each (batch, i, j) combination gets its own program
    with torch_device_fn.device(x1.device):
        grid = (
            n,  # pid_m = i (row index in result = row of x1)
            m,  # pid_n = j (col index in result = row of x2)
            batch_size,  # batch dimension
        )
        _cdist_forward_kernel[grid](
            x1,
            x2,
            out,
            n,
            m,
            d,
            p,
            x1.stride(0) if has_batch else 0,
            x1.stride(1) if has_batch else x1.stride(0),
            x1.stride(2) if has_batch else x1.stride(1),
            x2.stride(0) if has_batch else 0,
            x2.stride(1) if has_batch else x2.stride(0),
            x2.stride(2) if has_batch else x2.stride(1),
            out.stride(0) if has_batch else 0,
            out.stride(1) if has_batch else out.stride(0),
            out.stride(2) if has_batch else out.stride(1),
        )

    return out