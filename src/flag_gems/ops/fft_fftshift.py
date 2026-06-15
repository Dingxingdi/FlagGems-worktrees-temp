import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def fft_fftshift_impl(inp, dim=None):
    """Implementation of fft_fftshift that handles all cases."""
    # Handle dim parameter
    if dim is None:
        dims = list(range(inp.ndim))
    elif isinstance(dim, int):
        dims = [dim % inp.ndim]
    else:
        dims = [d % inp.ndim for d in dim]

    ndim = inp.ndim
    rank = ndim

    # Create shift amounts for all dimensions (0 if not shifting this dim)
    shifts = [0] * ndim
    for d in dims:
        shifts[d] = inp.shape[d] // 2

    # Build the kernel based on rank
    if rank == 1:
        return _fft_fftshift_1d(inp, shifts)
    elif rank == 2:
        return _fft_fftshift_2d(inp, shifts)
    elif rank == 3:
        return _fft_fftshift_3d(inp, shifts)
    elif rank == 4:
        return _fft_fftshift_4d(inp, shifts)
    else:
        # For higher ranks, fall back to torch (should be rare for FFT)
        return torch.fft.fftshift(inp, dim=dim)


@libentry()
@triton.heuristics({"BLOCK_M": lambda args: 128})
@triton.jit
def _fft_fftshift_1d_kernel(inp, out, N, shift0, BLOCK_M: tl.constexpr):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < N

    idx0 = (offsets + shift0) % N
    flat_idx = idx0

    # Triton interprets offset as element offset, so flat_idx is already correct
    val = tl.load(inp + flat_idx)
    tl.store(out + offsets, val, mask=mask)


def _fft_fftshift_1d(inp, shifts):
    logger.debug("GEMS FFT_FFTSHIFT_1D")
    N = inp.shape[0]
    shift0 = shifts[0]
    out = torch.empty_like(inp)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]),)
    _fft_fftshift_1d_kernel[grid](inp, out, N, shift0)
    return out


@libentry()
@triton.heuristics({"BLOCK_M": lambda args: 128})
@triton.jit
def _fft_fftshift_2d_kernel(inp, out, M, N, shift0, shift1, BLOCK_M: tl.constexpr):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M * N

    idx1 = offsets % N
    idx0 = (offsets // N) % M

    idx0 = (idx0 + shift0) % M
    idx1 = (idx1 + shift1) % N

    flat_idx = idx0 * N + idx1

    val = tl.load(inp + flat_idx)
    tl.store(out + offsets, val, mask=mask)


def _fft_fftshift_2d(inp, shifts):
    logger.debug("GEMS FFT_FFTSHIFT_2D")
    M, N = inp.shape
    shift0, shift1 = shifts[0], shifts[1]
    out = torch.empty_like(inp)

    total = M * N
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_M"]),)
    _fft_fftshift_2d_kernel[grid](inp, out, M, N, shift0, shift1)
    return out


@libentry()
@triton.heuristics({"BLOCK_M": lambda args: 128})
@triton.jit
def _fft_fftshift_3d_kernel(
    inp, out, M, N, K, shift0, shift1, shift2, BLOCK_M: tl.constexpr
):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M * N * K

    idx2 = offsets % K
    t = offsets // K
    idx1 = t % N
    idx0 = t // N

    idx0 = (idx0 + shift0) % M
    idx1 = (idx1 + shift1) % N
    idx2 = (idx2 + shift2) % K

    flat_idx = idx0 * N * K + idx1 * K + idx2

    val = tl.load(inp + flat_idx)
    tl.store(out + offsets, val, mask=mask)


def _fft_fftshift_3d(inp, shifts):
    logger.debug("GEMS FFT_FFTSHIFT_3D")
    M, N, K = inp.shape
    shift0, shift1, shift2 = shifts[0], shifts[1], shifts[2]
    out = torch.empty_like(inp)

    total = M * N * K
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_M"]),)
    _fft_fftshift_3d_kernel[grid](inp, out, M, N, K, shift0, shift1, shift2)
    return out


@libentry()
@triton.heuristics({"BLOCK_M": lambda args: 128})
@triton.jit
def _fft_fftshift_4d_kernel(
    inp, out, M, N, K, L, shift0, shift1, shift2, shift3, BLOCK_M: tl.constexpr
):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M * N * K * L

    idx3 = offsets % L
    t = offsets // L
    idx2 = t % K
    t = t // K
    idx1 = t % N
    idx0 = t // N

    idx0 = (idx0 + shift0) % M
    idx1 = (idx1 + shift1) % N
    idx2 = (idx2 + shift2) % K
    idx3 = (idx3 + shift3) % L

    flat_idx = idx0 * N * K * L + idx1 * K * L + idx2 * L + idx3

    val = tl.load(inp + flat_idx)
    tl.store(out + offsets, val, mask=mask)


def _fft_fftshift_4d(inp, shifts):
    logger.debug("GEMS FFT_FFTSHIFT_4D")
    M, N, K, L = inp.shape
    shift0, shift1, shift2, shift3 = shifts[0], shifts[1], shifts[2], shifts[3]
    out = torch.empty_like(inp)

    total = M * N * K * L
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_M"]),)
    _fft_fftshift_4d_kernel[grid](inp, out, M, N, K, L, shift0, shift1, shift2, shift3)
    return out


def fft_fftshift(inp: torch.Tensor, dim=None) -> torch.Tensor:
    logger.debug("GEMS FFT_FFTSHIFT")
    return fft_fftshift_impl(inp, dim)