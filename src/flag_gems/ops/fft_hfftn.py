"""
Implementation of fft_hfftn (half FFT for real input) for FlagGems.

This implements aten::fft_hfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor

The implementation uses Triton kernels for the FFT computation.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def complex_exp(angle):
    """Compute complex exponential: exp(i*angle) = cos(angle) + i*sin(angle)"""
    return tl.cos(angle), tl.sin(angle)


@triton.jit
def fft_hfftn_1d_dft_kernel(
    input_ptr,
    output_ptr_real,
    output_ptr_imag,
    stride_input,
    n_fft_points,
    n_fft_results,
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D DFT kernel for computing half FFT of real input.

    For each output frequency k (0 <= k <= N//2), compute:
    X[k] = sum(x[n] * exp(-2*pi*i*k*n/N)) for n = 0..N-1

    This is a naive O(N^2) implementation - suitable for small N or testing.
    """
    pid = tl.program_id(0)
    k = pid  # Which output frequency we're computing

    if k > n_fft_points // 2:
        return

    # Compute DFT: X[k] = sum(x[n] * exp(-2*pi*i*k*n/N))
    sum_real = 0.0
    sum_imag = 0.0

    # Loop over input elements
    for n in range(n_fft_points):
        # Load input element at position n
        x = tl.load(input_ptr + n * stride_input).to(tl.float32)

        # Twiddle factor: exp(-2*pi*i*k*n/N)
        angle = -2.0 * 3.141592653589793 * k * n / n_fft_points
        twiddle_real = tl.cos(angle)
        twiddle_imag = tl.sin(angle)

        # Accumulate
        sum_real += x * twiddle_real
        sum_imag += x * twiddle_imag

    # Store result
    output_idx = k
    tl.store(output_ptr_real + output_idx, sum_real)
    tl.store(output_ptr_imag + output_idx, sum_imag)


def _fft_hfftn_1d_triton(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Triton-based 1D half FFT implementation.

    Uses naive DFT for demonstration - production code would use FFT.
    """
    N = input.shape[dim]

    # Output size for half FFT of real input
    n_fft_results = N // 2 + 1

    # Prepare output tensors (real and imaginary parts separately for Triton)
    output_shape = list(input.shape)
    output_shape[dim] = n_fft_results

    output_real = torch.zeros(output_shape, dtype=torch.float32, device=input.device)
    output_imag = torch.zeros(output_shape, dtype=torch.float32, device=input.device)

    # For each slice along other dimensions, compute FFT
    # Flatten all dimensions except the FFT dim
    n_slices = input.numel() // N
    stride_input = input.stride(dim)

    # Launch kernel
    BLOCK_SIZE = 128
    grid = (n_slices * n_fft_results,)

    # Compute input and output strides for the flattened view
    # Each FFT is independent, so we process them in parallel

    # Actually, let's process each 1D FFT as a single task
    grid = (n_slices,)

    fft_hfftn_1d_dft_kernel[grid](
        input.contiguous().view(-1),
        output_real.contiguous().view(-1),
        output_imag.contiguous().view(-1),
        N,  # stride_input
        N,  # n_fft_points
        n_fft_results,
        BLOCK_SIZE,
    )

    # Combine real and imaginary parts
    output = torch.complex(output_real, output_imag)
    return output.view(output_shape)


def _fft_hfftn_impl(input: torch.Tensor, s=None, dim=None, norm=None) -> torch.Tensor:
    """
    Implementation of fft_hfftn.

    For real input, returns the "half" FFT - only the non-redundant half
    of the frequency components (first N//2 + 1 for each dimension).
    """
    # Handle default dim (all dimensions)
    if dim is None:
        dim = list(range(input.ndim))

    # Convert dim to list if it's a single int
    if isinstance(dim, int):
        dim = [dim]

    # Normalize dim values
    dim = [d if d >= 0 else d + input.ndim for d in dim]

    # Handle s (output shape) parameter
    if s is None:
        s = [input.shape[d] for d in dim]
    else:
        s = list(s)

    # Check if we can use Triton implementation
    # We support: single dimension FFT, power-of-2 or small sizes
    use_triton = False
    if len(dim) == 1 and s[0] <= 256:
        # Use Triton for small 1D FFTs
        use_triton = True

    if use_triton and len(dim) == 1:
        # Single dimension case - use Triton
        d = dim[0]
        N = s[0]

        # Pad input if needed
        if input.shape[d] != N:
            pad_config = [0] * (2 * input.ndim)
            pad_config[2 * (input.ndim - 1 - d)] = max(0, N - input.shape[d])
            input = torch.nn.functional.pad(input, pad_config)

        # Compute FFT using Triton kernel
        result = _fft_hfftn_1d_triton(input, dim=d)

        # Apply normalization
        if norm == "forward":
            result = result / N
        elif norm == "ortho":
            result = result / (N ** 0.5)

        return result
    else:
        # Fall back to torch for complex cases
        # Use torch's rfftn which computes half FFT for real input
        result = torch.fft.rfftn(input, s=s, dim=dim, norm=norm)
        return result


def fft_hfftn(
    input: torch.Tensor,
    s=None,
    dim=None,
    norm=None,
) -> torch.Tensor:
    """
    Compute the n-dimensional half FFT (also called real FFT).

    This is similar to torch.fft.fft but optimized for real-valued input,
    returning only the non-redundant half of the frequency components.

    Args:
        input: Input tensor (real-valued)
        s: Shape of the output tensor
        dim: Dimensions to transform
        norm: Normalization mode ("backward", "forward", "ortho", or None)

    Returns:
        Complex tensor containing the half FFT
    """
    logger.debug("GEMS fft_hfftn")
    return _fft_hfftn_impl(input, s=s, dim=dim, norm=norm)


def fft_hfftn_(
    input: torch.Tensor,
    s=None,
    dim=None,
    norm=None,
) -> torch.Tensor:
    """
    In-place version of fft_hfftn.
    """
    logger.debug("GEMS fft_hfftn_")
    result = _fft_hfftn_impl(input, s=s, dim=dim, norm=norm)
    input.copy_(result)
    return input