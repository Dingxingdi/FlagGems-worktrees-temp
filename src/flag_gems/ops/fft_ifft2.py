import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

PI = tl.constexpr(3.141592653589793)


@libentry()
@triton.jit
def fft2d_kernel(
    input_real_ptr,
    input_imag_ptr,
    output_real_ptr,
    output_imag_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    inverse: tl.constexpr,
):
    """2D FFT kernel for contiguous real/imag tensors."""
    pid = tl.program_id(0)
    i = pid // M  # row
    j = pid % M  # column

    acc_real = 0.0
    acc_imag = 0.0

    for k in range(N):
        for l in range(M):
            # Contiguous: row-major storage
            x_r = tl.load(input_real_ptr + k * M + l).to(tl.float32)
            x_i = tl.load(input_imag_ptr + k * M + l).to(tl.float32)

            # 2D DFT formula: exp(j*2*pi*(k*i/N + l*j/M))
            angle = 2.0 * PI * (k * i / N + l * j / M)
            if not inverse:
                angle = -angle

            w_r = tl_extra_shim.cos(angle)
            w_i = tl_extra_shim.sin(angle)

            prod_r = x_r * w_r - x_i * w_i
            prod_i = x_r * w_i + x_i * w_r

            acc_real += prod_r
            acc_imag += prod_i

    scale = 1.0 / (N * M) if inverse else 1.0
    out_idx = i * M + j

    tl.store(output_real_ptr + out_idx, acc_real * scale)
    tl.store(output_imag_ptr + out_idx, acc_imag * scale)


def fft_ifft2(
    input_tensor: torch.Tensor,
    s: torch.Size = None,
    dim: torch.Size = torch.Size([-2, -1]),
    norm: str = None,
) -> torch.Tensor:
    """2D inverse FFT.

    Args:
        input_tensor: Input complex tensor
        s: Output shape (optional)
        dim: Dimensions to apply FFT (default: [-2, -1])
        norm: Normalization mode ("forward", "backward", "ortho", or None)

    Returns:
        Complex tensor after inverse FFT
    """
    logger.debug("GEMS fft_ifft2")

    # Validate input
    if not input_tensor.is_complex():
        raise ValueError("fft_ifft2 requires complex input tensor")

    # Handle dim
    if isinstance(dim, (list, tuple)):
        dim = torch.Size(dim)
    elif isinstance(dim, torch.Size):
        pass
    else:
        dim = torch.Size([dim])

    # Default behavior: apply along last two dimensions
    if dim != torch.Size([-2, -1]):
        raise ValueError("fft_ifft2 currently only supports dim=[-2, -1]")

    # Get shapes for the last two dimensions
    if input_tensor.dim() >= 2:
        N = input_tensor.shape[-2]  # rows
        M = input_tensor.shape[-1]  # cols
    else:
        raise ValueError("Input tensor must be at least 2D")

    # Limit sizes for naive implementation
    if N * M > 64:
        raise ValueError(f"Currently only supports small FFT (N*M <= 64), got {N}x{M}")

    # Handle s (output shape)
    if s is not None:
        if not isinstance(s, torch.Size):
            s = torch.Size(s)

    # Get real and imaginary parts and make them contiguous
    real = input_tensor.real.contiguous()
    imag = input_tensor.imag.contiguous()

    # Prepare output tensors (contiguous)
    output_real = torch.zeros(N, M, dtype=real.dtype, device=real.device)
    output_imag = torch.zeros(N, M, dtype=imag.dtype, device=imag.device)

    # Handle batch dimensions
    if input_tensor.dim() == 2:
        # Simple 2D case
        grid = (N * M,)
        fft2d_kernel[grid](
            real,
            imag,
            output_real,
            output_imag,
            N,
            M,
            True,  # inverse
        )
    else:
        # Batch case - process each sample
        batch_size = input_tensor.shape[:-2]
        result = torch.empty_like(input_tensor)

        # Iterate over batch
        for idx in range(torch.tensor(batch_size).prod().item()):
            # Compute multi-dimensional index
            multi_idx = []
            remaining = idx
            for dim_size in batch_size:
                multi_idx.insert(0, remaining % dim_size)
                remaining = remaining // dim_size

            # Extract slice
            slice_obj = tuple(multi_idx + [slice(None), slice(None)])
            slice_data = input_tensor[slice_obj]

            # Process
            real_s = slice_data.real.contiguous()
            imag_s = slice_data.imag.contiguous()

            out_real_s = torch.zeros(N, M, dtype=real_s.dtype, device=real_s.device)
            out_imag_s = torch.zeros(N, M, dtype=imag_s.dtype, device=imag_s.device)

            fft2d_kernel[(N * M,)](
                real_s,
                imag_s,
                out_real_s,
                out_imag_s,
                N,
                M,
                True,
            )
            result[slice_obj] = torch.complex(out_real_s, out_imag_s)

        output_real = result.real
        output_imag = result.imag

    result = torch.complex(output_real, output_imag)

    # Handle normalization
    if norm == "forward":
        result = result / result.numel()
    elif norm == "ortho":
        result = result / math.sqrt(result.numel())

    # Handle output shape if specified
    if s is not None:
        result = result[..., : s[0], : s[1]]

    return result