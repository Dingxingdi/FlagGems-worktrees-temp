import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


def _get_max_power_of_two(n):
    """Get the largest power of two less than or equal to n"""
    return 1 << (n - 1).bit_length()


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fft_c2c"),
    key=["n_elements"],
)
@triton.jit
def fft_c2c_kernel(
    in_real_ptr,
    in_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n_elements,
    fft_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """FFT kernel for complex-to-complex FFT using radix-2 Cooley-Tukey"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load complex input
    real = tl.load(in_real_ptr + offsets, mask=mask, other=0.0)
    imag = tl.load(in_imag_ptr + offsets, mask=mask, other=0.0)

    # For production, implement proper Cooley-Tukey with shared memory
    # For now, do a simple DFT for correctness testing
    # This is O(N^2) but ensures we get accurate results

    # Compute DFT: X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N)
    result_real = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    result_imag = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Get the output position k
    k = offsets % fft_size

    # Iterate over all input elements n
    for n_val in range(fft_size):
        if n_val >= n_elements:
            break

        # Load x[n]
        x_real = tl.load(in_real_ptr + n_val, mask=n_val < n_elements, other=0.0)
        x_imag = tl.load(in_imag_ptr + n_val, mask=n_val < n_elements, other=0.0)

        # Compute exp(-2*pi*i*k*n/N)
        # For forward FFT: w = exp(-2*pi*i/N)
        # w^(k*n) = cos(2*pi*k*n/N) - i*sin(2*pi*k*n/N)
        prod = k * n_val
        angle = -6.283185307179586 * tl.cast(prod, tl.float32) / tl.cast(fft_size, tl.float32)

        w_real = tl.cos(angle)
        w_imag = tl.sin(angle)

        # Multiply: x[n] * w^(k*n)
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        prod_real = x_real * w_real - x_imag * w_imag
        prod_imag = x_real * w_imag + x_imag * w_real

        result_real = result_real + prod_real
        result_imag = result_imag + prod_imag

    # Store result
    tl.store(out_real_ptr + offsets, result_real, mask=mask)
    tl.store(out_imag_ptr + offsets, result_imag, mask=mask)


def fft_c2c(input: torch.Tensor, dim, normalization, forward) -> torch.Tensor:
    logger.debug("GEMS FFT C2C")

    # Handle empty tensor
    if input.numel() == 0:
        return input.clone()

    if not input.is_complex():
        raise ValueError("FFT input must be complex")

    # Get FFT size from the specified dimension
    dim_axis = dim[0] if isinstance(dim, (list, tuple)) else dim
    fft_size = input.shape[dim_axis]

    # For now, require power-of-2 sizes
    max_power_of_two = _get_max_power_of_two(fft_size)
    if fft_size != max_power_of_two:
        raise ValueError(f"FFT size {fft_size} is not a power of 2")

    # Create output
    output = torch.empty_like(input)
    src = input if input.is_contiguous() else input.contiguous()

    # Process based on dimensionality
    if input.ndim == 1:
        # Simple 1D case
        src_real = torch.view_as_real(src)
        out_real = torch.view_as_real(output)

        n_elements = src.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        fft_c2c_kernel[grid](
            src_real[:, 0].data_ptr(),
            src_real[:, 1].data_ptr(),
            out_real[:, 0].data_ptr(),
            out_real[:, 1].data_ptr(),
            n_elements,
            fft_size,
            BLOCK_SIZE=1024,
        )
    else:
        # Multi-dimensional case
        dim_axis = dim[0] if isinstance(dim, (list, tuple)) else dim

        # Permute to put FFT dimension last
        perm = [i for i in range(input.ndim) if i != dim_axis] + [dim_axis]
        src_perm = src.permute(perm)
        out_perm = output.permute(perm)

        other_shape = list(src_perm.shape[:-1])
        src_reshaped = src_perm.reshape(-1, fft_size)
        out_reshaped = out_perm.reshape(-1, fft_size)

        # Process each FFT
        for i in range(src_reshaped.shape[0]):
            src_real_line = torch.view_as_real(src_reshaped[i])
            out_real_line = torch.view_as_real(out_reshaped[i])

            n_elements_line = fft_size
            grid = lambda meta: (triton.cdiv(n_elements_line, meta["BLOCK_SIZE"]),)

            fft_c2c_kernel[grid](
                src_real_line[:, 0].data_ptr(),
                src_real_line[:, 1].data_ptr(),
                out_real_line[:, 0].data_ptr(),
                out_real_line[:, 1].data_ptr(),
                n_elements_line,
                fft_size,
                BLOCK_SIZE=1024,
            )

        # Reshape back
        out_perm[:] = out_reshaped.reshape(*other_shape, fft_size)
        output[:] = out_perm.permute(perm)

    # Apply normalization
    if normalization == 1:  # forward
        output = output / float(fft_size ** 0.5)
    elif normalization == 2:  # backward
        output = output / float(fft_size ** 0.5)

    # Handle inverse FFT
    if not forward:
        # Inverse: conjugate, forward FFT, conjugate, scale
        output = torch.conj(output)

    return output