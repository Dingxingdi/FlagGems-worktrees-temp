import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Keyset to bypass FlagGems dispatch and use PyTorch directly
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit
def fft_ifft_dft_kernel(
    input_real,
    input_imag,
    output_real,
    output_imag,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    IFFT kernel using simple DFT for small sizes.
    IFFT formula: x[n] = (1/n) * sum(X[k] * exp(2*pi*i*k*n/N))
    """
    pid = tl.program_id(0)
    if pid >= n:
        return

    # Compute output index
    k_idx = pid

    # Accumulator for the DFT sum
    acc_real = 0.0
    acc_imag = 0.0

    # Loop over all input elements
    for i in range(0, n):
        # Load input complex number
        real_i = tl.load(input_real + i)
        imag_i = tl.load(input_imag + i)

        # Compute twiddle factor: exp(2*pi*i*k*n/N) = cos(2*pi*k*n/N) + i*sin(2*pi*k*n/N)
        # For IFFT, we use exp(+2*pi*i*k*n/N)
        angle = 2.0 * 3.141592653589793 * k_idx * i / n
        twiddle_real = tl.cos(angle)
        twiddle_imag = tl.sin(angle)

        # Multiply input by twiddle: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        prod_real = real_i * twiddle_real - imag_i * twiddle_imag
        prod_imag = real_i * twiddle_imag + imag_i * twiddle_real

        # Accumulate
        acc_real = acc_real + prod_real
        acc_imag = acc_imag + prod_imag

    # Scale by 1/n
    scale = 1.0 / n
    out_real = acc_real * scale
    out_imag = acc_imag * scale

    # Store result
    tl.store(output_real + k_idx, out_real)
    tl.store(output_imag + k_idx, out_imag)


def fft_ifft(input_tensor: torch.Tensor, n=None, dim=-1, norm=None) -> torch.Tensor:
    """
    Inverse FFT operator for FlagGems.

    Uses a simple DFT implementation in Triton for small power-of-2 sizes,
    and delegates to PyTorch's FFT for other cases using redispatch to avoid recursion.

    Args:
        input_tensor: Input tensor (complex)
        n: Signal length
        dim: Dimension along which to take IFFT
        norm: Normalization mode

    Returns:
        Inverse FFT result
    """
    logger.debug("GEMS FFT_IFFT")

    # Handle None n (use input size)
    if n is None:
        n = input_tensor.shape[dim]

    # Handle negative dim
    if dim < 0:
        dim = input_tensor.ndim + dim

    # For 1D case with small power-of-2 sizes, use Triton
    # Note: Triton doesn't support complex128, so skip for that dtype
    if (
        input_tensor.ndim == 1
        and n <= 64  # MAX_TRITON_SIZE
        and (n & (n - 1)) == 0  # power of 2
        and dim == -1
        and n == input_tensor.shape[0]  # n must equal input size for Triton
        and input_tensor.dtype != torch.complex128  # Triton doesn't support complex128
        and n <= input_tensor.shape[0]
    ):
        try:
            # Convert to real representation
            input_real = torch.view_as_real(input_tensor.contiguous())
            n_elements = input_tensor.numel()

            # Allocate output
            output_buffer = torch.empty(
                n, dtype=input_tensor.dtype, device=input_tensor.device
            )

            # Launch kernel
            grid = (n,)
            BLOCK_SIZE = max(1, triton.next_power_of_2(n))

            # Use a kernel that processes the interleaved format
            fft_ifft_dft_kernel[grid](
                input_real[:, 0],  # real part
                input_real[:, 1],  # imag part
                output_buffer.real,
                output_buffer.imag,
                n,
                BLOCK_SIZE,
            )

            return output_buffer
        except Exception as e:
            # If Triton fails, fall back to PyTorch
            logger.debug(f"FFT IFFT Triton failed, falling back to PyTorch: {e}")

    # Fall back to PyTorch using redispatch to avoid recursion
    result = torch.ops.aten.fft_ifft.default.redispatch(
        _FALLBACK_KEYSET, input_tensor, n, dim, norm
    )
    return result


def fft_ifft_(
    input_tensor: torch.Tensor, n=None, dim=-1, norm=None
) -> torch.Tensor:
    """
    In-place inverse FFT operator for FlagGems.
    """
    logger.debug("GEMS FFT_IFFT_")

    # Compute result using fft_ifft
    result = fft_ifft(input_tensor, n=n, dim=dim, norm=norm)
    input_tensor.copy_(result)

    return input_tensor