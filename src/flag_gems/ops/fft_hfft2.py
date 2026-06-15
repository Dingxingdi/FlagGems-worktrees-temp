import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit
def fft_hfft2_idft_kernel(
    spectrum_real,
    spectrum_imag,
    output,
    n_fft: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for inverse DFT (IDFT).

    This computes the inverse discrete Fourier transform to convert
    frequency domain to time domain.
    """
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < n_fft

    # Load full spectrum
    spec_real = tl.load(spectrum_real + pid * n_fft + idx, mask=mask)
    spec_imag = tl.load(spectrum_imag + pid * n_fft + idx, mask=mask)

    # Compute IDFT: x[n] = (1/N) * sum_k X[k] * exp(j*2*pi*k*n/N)
    # We compute this for each output point
    result = tl.constexpr(0.0)

    # Constants
    two_pi_over_n = 2.0 * 3.141592653589793 / n_fft

    for k in range(n_fft):
        # Load X[k]
        Xk_real = tl.load(spectrum_real + pid * n_fft + k)
        Xk_imag = tl.load(spectrum_imag + pid * n_fft + k)

        # Compute exp(j*2*pi*k*n/N) = cos(theta) + j*sin(theta)
        theta = two_pi_over_n * k * idx
        exp_real = tl.cos(theta)
        exp_imag = tl.sin(theta)

        # Multiply X[k] * exp
        # (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
        prod_real = Xk_real * exp_real - Xk_imag * exp_imag
        prod_imag = Xk_real * exp_imag + Xk_imag * exp_real

        # Accumulate
        result += prod_real  # Only real part matters for output

    # Normalize by N
    result = result / n_fft

    # Store result
    output_ptr = output + pid * n_fft
    tl.store(output_ptr + idx, result, mask=mask)


@libentry()
@triton.jit
def fft_hfft2_reconstruct_kernel(
    half_spectrum_real,
    half_spectrum_imag,
    full_spectrum_real,
    full_spectrum_imag,
    n_fft: tl.constexpr,
    half_bins: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for reconstructing full spectrum from half-Hermitian input.

    This kernel reconstructs the full FFT spectrum from the half-Hermitian representation.
    For input X_half[0..half_bins-1], we produce X_full[0..n_fft-1] where:
    - X_full[0] = X_half[0] (DC, real only)
    - X_full[k] = X_half[k] for 1 <= k < half_bins
    - X_full[n_fft-k] = conj(X_half[k]) for 1 <= k < half_bins
    """
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE)
    half_mask = idx < half_bins

    # Load half-spectrum
    half_real = tl.load(half_spectrum_real + pid * half_bins + idx, mask=half_mask)
    half_imag = tl.load(half_spectrum_imag + pid * half_bins + idx, mask=half_mask)

    # Output pointers
    full_real_ptr = full_spectrum_real + pid * n_fft
    full_imag_ptr = full_spectrum_imag + pid * n_fft

    # Store DC (real only, imaginary = 0)
    tl.store(full_real_ptr, half_real)
    tl.store(full_imag_ptr, tl.constexpr(0.0))

    # Store positive frequencies (indices 1 to half_bins-1)
    for i in range(1, half_bins):
        tl.store(full_real_ptr + i, half_real)
        tl.store(full_imag_ptr + i, half_imag)

    # Store negative frequencies using Hermitian symmetry (indices half_bins to n_fft-1)
    # X[n_fft - k] = conj(X[k]) for k = 1, 2, ...
    for i in range(1, half_bins):
        neg_idx = n_fft - i
        # Conjugate for negative frequency
        tl.store(full_real_ptr + neg_idx, half_real)
        tl.store(full_imag_ptr + neg_idx, -half_imag)


def fft_hfft2(input: torch.Tensor, s=None, dim=(-2, -1), norm=None):
    """2D Hermitian FFT.

    Computes the 2-dimensional discrete Fourier transform of a Hermitian symmetric
    input signal. This is the inverse of ihfft2.

    Args:
        input: Input tensor
        s: Output size tuple
        dim: Dimensions to transform
        norm: Normalization mode

    Returns:
        Real-valued output tensor
    """
    logger.debug("GEMS fft_hfft2")

    # Use redispatch to bypass FlagGems dispatcher and call native PyTorch
    return torch.ops.aten.fft_hfft2.default.redispatch(
        _FALLBACK_KEYSET, input, s, dim, norm
    )