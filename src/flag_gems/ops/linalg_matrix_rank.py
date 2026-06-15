import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def count_rank_kernel(
    svals_ptr,
    out_ptr,
    num_sv: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_sv

    svals = tl.load(svals_ptr + offset, mask=mask, other=0.0)
    # Count singular values greater than threshold
    count = tl.sum(tl.where(svals > threshold, 1, 0))

    tl.store(out_ptr + pid, count)


def linalg_matrix_rank(A, tol=None, hermitian=False):
    logger.debug("GEMS linalg_matrix_rank")

    # Get input shape
    shape = A.shape
    if len(shape) < 2:
        raise ValueError("linalg_matrix_rank: expected input tensor with at least 2 dimensions")

    m, n = shape[-2], shape[-1]
    batch_dims = shape[:-2]
    batch_size = math.prod(batch_dims) if batch_dims else 1

    # Compute singular values using torch
    # svals has shape (*batch_dims, min(m, n))
    singular_values = torch.linalg.svdvals(A)

    # Determine threshold
    if tol is None:
        # Default threshold: max(m, n) * eps
        eps = torch.finfo(A.dtype).eps
        threshold = max(m, n) * eps
    else:
        threshold = tol

    # For batched inputs, we need to compute rank for each matrix
    if batch_size == 1:
        # Single matrix case
        svals = singular_values.reshape(-1)
        num_sv = len(svals)

        # Use Triton kernel to count
        BLOCK_SIZE = triton.next_power_of_2(num_sv)
        out = torch.empty(1, dtype=torch.int64, device=A.device)

        with torch_device_fn.device(A.device):
            count_rank_kernel[(1, 1, 1)](
                svals,
                out,
                num_sv,
                threshold,
                BLOCK_SIZE,
            )
        return out.squeeze()
    else:
        # Batched case: compute rank for each matrix in the batch
        out = torch.empty(batch_dims, dtype=torch.int64, device=A.device)
        min_dim = min(m, n)

        BLOCK_SIZE = triton.next_power_of_2(min_dim)

        # Reshape for batch processing
        svals_flat = singular_values.reshape(batch_size, min_dim)

        with torch_device_fn.device(A.device):
            count_rank_kernel[(batch_size, 1, 1)](
                svals_flat,
                out,
                min_dim,
                threshold,
                BLOCK_SIZE,
            )

        # Reshape output to match batch dimensions
        return out.reshape(batch_dims)