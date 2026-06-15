import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics({})
@triton.jit
def fft_rfft2_row_kernel(
    input_ptr,
    intermediate_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: Compute 1D FFT on each row.
    For real input, we compute full FFT then take only positive frequencies.
    """
    row_idx = tl.program_id(0)
    freq_idx = tl.program_id(1)

    if row_idx >= n_rows:
        return

    n_output = n_cols // 2 + 1
    if freq_idx >= n_output:
        return

    k = freq_idx

    real_sum = 0.0
    imag_sum = 0.0

    # Compute 1D FFT for this row at frequency k
    for n in range(n_cols):
        offset = row_idx * n_cols + n
        x_n = tl.load(input_ptr + offset)

        angle = -6.283185307179586 * k * n / n_cols
        twiddle_real = tl.math.cos(angle)
        twiddle_imag = tl.math.sin(angle)

        real_sum = real_sum + x_n * twiddle_real
        imag_sum = imag_sum + x_n * twiddle_imag

    # Store complex result (as interleaved)
    # Intermediates are stored as [row, freq, real/imag]
    intermediate_idx = row_idx * n_output + freq_idx
    tl.store(intermediate_ptr + intermediate_idx * 2, real_sum)
    tl.store(intermediate_ptr + intermediate_idx * 2 + 1, imag_sum)


@libentry()
@triton.heuristics({})
@triton.jit
def fft_rfft2_col_kernel(
    intermediate_ptr,
    output_ptr,
    n_rows,
    n_cols,
    output_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2: Compute 1D FFT on each column of the row-FFT result.
    The intermediate is already complex (from row FFT), so we do full FFT on columns.
    """
    # For output[row, col], we need to compute column FFT at that column
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    if row_idx >= n_rows:
        return

    if col_idx >= output_cols:
        return

    p = row_idx  # row frequency
    k = col_idx  # column frequency

    real_sum = 0.0
    imag_sum = 0.0

    # Compute FFT along columns
    # For each row m in the intermediate result
    for m in range(n_rows):
        # Load complex value from intermediate[m, k]
        m_idx = m * output_cols + col_idx
        real_m = tl.load(intermediate_ptr + m_idx * 2)
        imag_m = tl.load(intermediate_ptr + m_idx * 2 + 1)

        # Twiddle factor for column FFT: exp(-2*pi*i*p*m/n_rows)
        angle = -6.283185307179586 * p * m / n_rows
        twiddle_real = tl.math.cos(angle)
        twiddle_imag = tl.math.sin(angle)

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_sum = real_sum + real_m * twiddle_real - imag_m * twiddle_imag
        imag_sum = imag_sum + real_m * twiddle_imag + imag_m * twiddle_real

    # Store result
    output_idx = row_idx * output_cols + col_idx
    tl.store(output_ptr + output_idx * 2, real_sum)
    tl.store(output_ptr + output_idx * 2 + 1, imag_sum)


def fft_rfft2(
    input: torch.Tensor,
    s=None,
    dim=(-2, -1),
    norm="backward",
    out=None,
):
    """
    Compute 2D FFT for real input using two-pass approach:
    1. Row FFT (rfft per row)
    2. Column FFT (fft per column)
    """
    logger.debug("GEMS FFT_RFFT2")

    # Handle default dimensions
    if dim == (-2, -1) or dim == (-1, -2):
        dim = (-2, -1)

    # Get input shape
    n_rows = input.shape[-2]
    n_cols = input.shape[-1]

    # Handle s parameter for output size
    if s is not None:
        if len(s) >= 2:
            n_rows = s[-2]
            n_cols = s[-1]
        elif len(s) == 1:
            n_cols = s[0]

    # Output size for rfft2: n_cols//2 + 1
    output_cols = n_cols // 2 + 1

    # Phase 1: Row FFT
    # Output is complex: (n_rows, output_cols)
    row_fft_shape = list(input.shape[:-1]) + [output_cols]
    row_fft_intermediate = torch.empty(
        row_fft_shape + [2],
        dtype=input.dtype,
        device=input.device,
    )

    grid1 = (n_rows, output_cols)
    fft_rfft2_row_kernel[grid1](
        input,
        row_fft_intermediate,
        n_rows,
        n_cols,
        128,  # BLOCK_SIZE
    )

    # Phase 2: Column FFT on row FFT results
    # Output is complex: (n_rows, output_cols)
    output_shape = list(input.shape[:-1]) + [output_cols]
    output_interleaved = torch.empty(
        output_shape + [2],
        dtype=input.dtype,
        device=input.device,
    )

    grid2 = (n_rows, output_cols)
    fft_rfft2_col_kernel[grid2](
        row_fft_intermediate,
        output_interleaved,
        n_rows,
        n_cols,
        output_cols,
        128,  # BLOCK_SIZE
    )

    # Convert interleaved to complex with appropriate dtype
    # For float16 input -> complex32 output, float32 -> complex64
    if input.dtype == torch.float16:
        output = torch.view_as_complex(output_interleaved.half())
    else:
        output = torch.view_as_complex(output_interleaved.float())

    # Apply normalization
    if norm == "forward":
        n = n_rows * n_cols
        output = output / n
    elif norm == "ortho":
        n = n_rows * n_cols
        output = output / (n ** 0.5)
    # "backward" - no normalization (default)

    return output


def fft_rfft2_(
    input: torch.Tensor,
    s=None,
    dim=(-2, -1),
    norm="backward",
):
    """
    In-place 2D FFT for real input.
    """
    logger.debug("GEMS FFT_RFFT2_")
    return fft_rfft2(input, s=s, dim=dim, norm=norm)