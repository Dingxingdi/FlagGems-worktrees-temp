import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _stft_kernel(
    input_ptr,
    window_ptr,
    output_real_ptr,
    output_imag_ptr,
    input_len: tl.constexpr,
    n_fft: tl.constexpr,
    win_length: tl.constexpr,
    hop_length: tl.constexpr,
    n_frames: tl.constexpr,
    n_freq_bins: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    FREQ_STRIDE: tl.constexpr,
    FRAME_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    STFT kernel using naive DFT computation.
    Each program instance computes one (batch, freq) pair for all frames.
    """
    # Get the frequency index for this program
    pid = tl.program_id(0)
    batch_idx = pid // n_freq_bins
    freq_idx = pid % n_freq_bins

    if freq_idx >= n_freq_bins:
        return

    # Compute 2*pi*freq_idx/n_fft
    two_pi_over_n = 2.0 * 3.141592653589793 / n_fft
    omega = two_pi_over_n * freq_idx

    # Input and output pointers for this batch
    input_ptr_batch = input_ptr + batch_idx * BATCH_STRIDE
    output_real_ptr_batch = output_real_ptr + batch_idx * FREQ_STRIDE
    output_imag_ptr_batch = output_imag_ptr + batch_idx * FREQ_STRIDE

    # Compute DFT for each frame
    for frame_idx in range(n_frames):
        # Calculate start position in input
        start_pos = frame_idx * hop_length

        # Accumulator using reduction
        sum_real = 0.0
        sum_imag = 0.0

        # Process all elements in win_length using vectorized loads
        # We process up to BLOCK_SIZE elements at a time
        num_blocks = (win_length + BLOCK_SIZE - 1) // BLOCK_SIZE

        for block_id in range(num_blocks):
            # Create offsets for this block
            base_offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = base_offs < win_length

            # Load window values
            window_vals = tl.load(window_ptr + base_offs, mask=mask, other=0.0)

            # Calculate input positions
            input_pos = start_pos + base_offs
            # Clamp to valid range [0, input_len)
            input_pos = tl.where(input_pos >= 0, input_pos, 0)
            input_pos = tl.where(input_pos < input_len, input_pos, input_len - 1)

            # Load input values
            x = tl.load(input_ptr_batch + input_pos, mask=mask, other=0.0)

            # Apply window
            x_windowed = x * window_vals

            # Compute DFT components - vectorized
            phase = omega * base_offs.to(tl.float32)
            cos_val = tl.cos(phase)
            sin_val = tl.sin(phase)

            # Sum across the block - use reduce instead of sum
            block_real = tl.sum(x_windowed * cos_val)
            block_imag = tl.sum(-x_windowed * sin_val)

            sum_real = sum_real + block_real
            sum_imag = sum_imag + block_imag

        # Store output
        tl.store(output_real_ptr_batch + frame_idx * FRAME_STRIDE + freq_idx, sum_real)
        tl.store(output_imag_ptr_batch + frame_idx * FRAME_STRIDE + freq_idx, sum_imag)


def stft(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int = None,
    win_length: int = None,
    window: torch.Tensor = None,
    center: bool = False,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool = None,
    return_complex: bool = None,
):
    """Short-Time Fourier Transform.

    This is a simplified implementation that supports the most common use cases.
    """
    logger.debug("GEMS STFT")

    # Handle defaults
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if onesided is None:
        onesided = True
    if return_complex is None:
        return_complex = not input.is_complex()

    # Store original shape
    original_shape = input.shape

    # Handle 1D input (add batch dimension if needed)
    original_dim = input.dim()
    if input.dim() == 1:
        input = input.unsqueeze(0)

    batch_size = input.shape[0]
    input_len = input.shape[1]

    # Pad input if center=True
    if center:
        pad = n_fft // 2
        input = torch.nn.functional.pad(input, (pad, pad), mode=pad_mode)
        input_len = input.shape[1]

    # Calculate number of frames
    n_frames = 1 + (input_len - win_length) // hop_length

    # Output frequency bins
    if onesided:
        n_freq_bins = n_fft // 2 + 1
    else:
        n_freq_bins = n_fft

    # Handle window - default to rectangular window (all ones) like PyTorch
    if window is None:
        window = torch.ones(win_length, dtype=input.dtype, device=input.device)
    else:
        window = window.to(dtype=input.dtype, device=input.device)

    # Ensure input is contiguous
    input = input.contiguous()

    # Allocate output buffers for real and imaginary parts
    # Use layout [batch, frame, freq] for easier kernel access
    real_out = torch.zeros(
        (batch_size, n_frames, n_freq_bins),
        dtype=input.dtype,
        device=input.device,
    )
    imag_out = torch.zeros(
        (batch_size, n_frames, n_freq_bins),
        dtype=input.dtype,
        device=input.device,
    )

    # Grid: (batch_size * n_freq_bins,)
    grid = (batch_size * n_freq_bins,)

    BLOCK_SIZE = 512

    # Launch kernel
    with torch_device_fn.device(input.device):
        _stft_kernel[grid](
            input,
            window,
            real_out,
            imag_out,
            input_len,
            n_fft,
            win_length,
            hop_length,
            n_frames,
            n_freq_bins,
            input.stride(0),
            real_out.stride(0),
            real_out.stride(1),
            BLOCK_SIZE,
        )

    # Combine real and imaginary into output
    # Output shape from kernel: [batch, frame, freq]
    # Final output shape: [batch, freq, frames]
    if return_complex:
        # Transpose from [batch, frame, freq] to [batch, freq, frame]
        real_out_t = real_out.transpose(1, 2)
        imag_out_t = imag_out.transpose(1, 2)
        output = torch.complex(real_out_t, imag_out_t)
        # Handle normalization for complex tensors
        if normalized:
            output = output * float(n_fft**-0.5)
    else:
        # Return as real tensor with extra dimension for real/imag
        # Transpose from [batch, frame, freq] to [batch, freq, frame]
        real_out_t = real_out.transpose(1, 2)
        imag_out_t = imag_out.transpose(1, 2)
        output = torch.empty(
            (batch_size, n_freq_bins, n_frames, 2),
            dtype=input.dtype,
            device=input.device,
        )
        output[:, :, :, 0] = real_out_t
        output[:, :, :, 1] = imag_out_t
        # Handle normalization for real tensors
        if normalized:
            output = output * float(n_fft**-0.5)

    # Restore original dimensions
    if original_dim == 1:
        output = output.squeeze(0)

    return output


def stft_(
    input: torch.Tensor,
    n_fft: int,
    hop_length: int = None,
    win_length: int = None,
    window: torch.Tensor = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: bool = None,
    return_complex: bool = None,
):
    """In-place STFT - not supported, delegates to stft."""
    logger.debug("GEMS STFT_")
    return stft(
        input,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
        return_complex,
    )