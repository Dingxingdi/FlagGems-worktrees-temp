import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def cholesky_solve_kernel(
    B_ptr,
    L_ptr,
    output_ptr,
    n,
    k,
    stride_bb,
    stride_bn,
    stride_bk,
    stride_lb,
    stride_ln,
    stride_lk,
    stride_ob,
    stride_on,
    stride_ok,
    upper: tl.constexpr,
):
    # Each program processes one batch element
    pid = tl.program_id(0)

    # Move pointers to the right batch
    B_ptr = B_ptr + pid * stride_bb
    L_ptr = L_ptr + pid * stride_lb
    output_ptr = output_ptr + pid * stride_ob

    # Initialize output = B (convert to float32 for computation)
    for i in range(n):
        for j in range(k):
            b_val = tl.load(B_ptr + i * stride_bn + j * stride_bk).to(tl.float32)
            tl.store(output_ptr + i * stride_on + j * stride_ok, b_val)

    # Forward substitution: L @ Y = B
    for i in range(n):
        for j in range(k):
            y_ptr = output_ptr + i * stride_on + j * stride_ok
            y_ij = tl.load(y_ptr).to(tl.float32)

            if upper:
                # Using U^T
                for jj in range(i):
                    l_ji = tl.load(L_ptr + jj * stride_ln + i * stride_lk).to(tl.float32)
                    y_jj = tl.load(output_ptr + jj * stride_on + j * stride_ok).to(tl.float32)
                    y_ij = y_ij - l_ji * y_jj
                l_ii = tl.load(L_ptr + i * stride_ln + i * stride_lk).to(tl.float32)
            else:
                # Using L
                for jj in range(i):
                    l_ij = tl.load(L_ptr + i * stride_ln + jj * stride_lk).to(tl.float32)
                    y_jj = tl.load(output_ptr + jj * stride_on + j * stride_ok).to(tl.float32)
                    y_ij = y_ij - l_ij * y_jj
                l_ii = tl.load(L_ptr + i * stride_ln + i * stride_lk).to(tl.float32)

            y_ij = y_ij / l_ii
            tl.store(y_ptr, y_ij)

    # Backward substitution: L^T @ X = Y
    for i in range(n - 1, -1, -1):
        for j in range(k):
            x_ptr = output_ptr + i * stride_on + j * stride_ok
            x_ij = tl.load(x_ptr).to(tl.float32)

            if upper:
                # Using U
                for jj in range(i + 1, n):
                    l_ij = tl.load(L_ptr + i * stride_ln + jj * stride_lk).to(tl.float32)
                    x_jj = tl.load(output_ptr + jj * stride_on + j * stride_ok).to(tl.float32)
                    x_ij = x_ij - l_ij * x_jj
                l_ii = tl.load(L_ptr + i * stride_ln + i * stride_lk).to(tl.float32)
            else:
                # Using L^T
                for jj in range(i + 1, n):
                    l_ji = tl.load(L_ptr + jj * stride_ln + i * stride_lk).to(tl.float32)
                    x_jj = tl.load(output_ptr + jj * stride_on + j * stride_ok).to(tl.float32)
                    x_ij = x_ij - l_ji * x_jj
                l_ii = tl.load(L_ptr + i * stride_ln + i * stride_lk).to(tl.float32)

            x_ij = x_ij / l_ii
            tl.store(x_ptr, x_ij)


def cholesky_solve(B: torch.Tensor, L: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """
    Solve the system of linear equations using Cholesky decomposition.

    Args:
        B: Right-hand side tensor of shape (*, n, k)
        L: Cholesky decomposition tensor of shape (*, n, n)
        upper: Whether L is upper triangular (True) or lower triangular (False)

    Returns:
        Solution tensor of shape (*, n, k)
    """
    logger.debug("GEMS cholesky_solve")

    # Get dimensions
    batch_dims = B.shape[:-2]
    n = B.shape[-2]
    k = B.shape[-1]

    # Handle batch dimensions
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    # Ensure inputs are contiguous
    L = L.contiguous()
    B = B.contiguous()

    # Output tensor
    output = torch.empty_like(B)

    # Compute grid - one program per batch element
    grid = (batch_size,)

    with torch_device_fn.device(B.device):
        cholesky_solve_kernel[grid](
            B,
            L,
            output,
            n,
            k,
            B.stride(-3) if B.dim() >= 3 else 0,
            B.stride(-2),
            B.stride(-1),
            L.stride(-3) if L.dim() >= 3 else 0,
            L.stride(-2),
            L.stride(-1),
            output.stride(-3) if output.dim() >= 3 else 0,
            output.stride(-2),
            output.stride(-1),
            upper,
            num_warps=4,
        )

    return output