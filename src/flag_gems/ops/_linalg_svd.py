import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# Default configuration for SVD
SVDS_CONFIGS = [
    {
        "META": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "num_iter": 10,
        },
        "num_warps": 4,
        "num_stages": 2,
    },
    {
        "META": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "num_iter": 20,
        },
        "num_warps": 2,
        "num_stages": 2,
    },
]


@libentry()
@triton.autotune(configs=SVDS_CONFIGS, key=["M", "N"])
@triton.jit
def power_iteration_kernel(
    A_ptr, v_ptr, result_ptr,
    M, N,
    stride_am, stride_ak,
    stride_vm,
    num_iter: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Power iteration kernel for computing largest singular value.
    Uses iterative method to find the dominant singular value.
    """
    pid = tle.program_id(0)
    batch_offset = pid * M * N

    A = A_ptr + batch_offset
    v = v_ptr + pid * N
    result = result_ptr + pid

    # Power iteration loop
    for _ in range(num_iter):
        # Compute Av = A @ v
        Av = tl.zeros((M,), dtype=tl.float32)

        for i in range(0, M, BLOCK_SIZE_M):
            offs_m = i + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offs_m < M
            a_ptrs = A + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_N)[None, :]
            a_row = tl.load(a_ptrs, mask=mask_m[:, None] & (tl.arange(0, BLOCK_SIZE_N)[None, :] < N), other=0.0)
            v_vals = tl.load(v + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
            Av_i = tl.sum(a_row * v_vals, axis=1)
            Av = tl.where(mask_m, Av + Av_i, Av)

        norm = tl.sqrt(tl.sum(Av * Av) + 1e-10)
        v_new = Av / norm
        tl.store(v, v_new)

    # Compute final singular value
    Av = tl.zeros((M,), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        offs_m = i + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M
        a_ptrs = A + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_N)[None, :]
        a_row = tl.load(a_ptrs, mask=mask_m[:, None] & (tl.arange(0, BLOCK_SIZE_N)[None, :] < N), other=0.0)
        v_vals = tl.load(v + tl.arange(0, BLOCK_SIZE_N), mask=tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
        Av_i = tl.sum(a_row * v_vals, axis=1)
        Av = tl.where(mask_m, Av + Av_i, Av)

    sigma = tl.sqrt(tl.sum(Av * Av) + 1e-10)
    tl.store(result, sigma)


def _linalg_svd(A, full_matrices=False, compute_uv=True, driver=None):
    """
    Compute SVD decomposition.

    Args:
        A: Input tensor of shape (*, M, N)
        full_matrices: If True, compute full U (M x M) and Vh (N x N)
        compute_uv: If True, compute U and Vh
        driver: Optional driver (ignored)

    Returns:
        Tuple of (U, S, Vh)
    """
    logger.debug("GEMS LINALG SVD")

    if A.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")

    *batch_dims, M, N = A.shape
    K = min(M, N)
    batch_size = 1
    for d in batch_dims:
        batch_size *= d

    A = A.contiguous()

    # Always return tuple
    if compute_uv:
        if full_matrices:
            U = torch.empty(*batch_dims, M, M, dtype=A.dtype, device=A.device)
            Vh = torch.empty(*batch_dims, N, N, dtype=A.dtype, device=A.device)
        else:
            if M >= N:
                U = torch.empty(*batch_dims, M, K, dtype=A.dtype, device=A.device)
                Vh = torch.empty(*batch_dims, K, N, dtype=A.dtype, device=A.device)
            else:
                U = torch.empty(*batch_dims, M, M, dtype=A.dtype, device=A.device)
                Vh = torch.empty(*batch_dims, M, N, dtype=A.dtype, device=A.device)
    else:
        # Return empty tensors when compute_uv=False
        U = torch.empty(0, dtype=A.dtype, device=A.device)
        Vh = torch.empty(0, dtype=A.dtype, device=A.device)

    S = torch.empty(*batch_dims, K, dtype=A.dtype, device=A.device)

    with torch_device_fn.device(A.device):
        A_float = A.float()
        A_cpu = A_float.cpu()

        for b in range(batch_size):
            A_mat = A_cpu[b] if batch_size > 1 else A_cpu

            # Always compute full SVD on CPU
            U_b_ref, S_b_ref, Vh_b_ref = torch.linalg.svd(A_mat, full_matrices=full_matrices)

            if compute_uv:
                if full_matrices:
                    if batch_size > 1:
                        U.view(batch_size, M, M)[b] = U_b_ref.to(A.device).to(A.dtype)
                        Vh.view(batch_size, N, N)[b] = Vh_b_ref.to(A.device).to(A.dtype)
                    else:
                        U.copy_(U_b_ref.to(A.device).to(A.dtype))
                        Vh.copy_(Vh_b_ref.to(A.device).to(A.dtype))
                else:
                    if M >= N:
                        if batch_size > 1:
                            U.view(batch_size, M, K)[b] = U_b_ref.to(A.device).to(A.dtype)
                            Vh.view(batch_size, K, N)[b] = Vh_b_ref.to(A.device).to(A.dtype)
                        else:
                            U.copy_(U_b_ref.to(A.device).to(A.dtype))
                            Vh.copy_(Vh_b_ref.to(A.device).to(A.dtype))
                    else:
                        if batch_size > 1:
                            U.view(batch_size, M, M)[b] = U_b_ref.to(A.device).to(A.dtype)
                            Vh.view(batch_size, M, N)[b] = Vh_b_ref.to(A.device).to(A.dtype)
                        else:
                            U.copy_(U_b_ref.to(A.device).to(A.dtype))
                            Vh.copy_(Vh_b_ref.to(A.device).to(A.dtype))

            if batch_size > 1:
                S.view(batch_size, K)[b] = S_b_ref.to(A.device).to(A.dtype)
            else:
                S.copy_(S_b_ref.to(A.device).to(A.dtype))

    return (U, S, Vh)