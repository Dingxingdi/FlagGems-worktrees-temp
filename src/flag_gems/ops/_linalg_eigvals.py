import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)


@triton.jit
def solve_2x2_eigenvalues(a00, a01, a10, a11):
    # Characteristic equation: lambda^2 - tr(A)*lambda + det(A) = 0
    # lambda = (tr +/- sqrt(tr^2 - 4*det)) / 2
    trace = a00 + a11
    det = a00 * a11 - a01 * a10
    discriminant = trace * trace - 4.0 * det

    # For complex eigenvalues (discriminant < 0):
    # real part = trace / 2
    # imag part = sqrt(|discriminant|) / 2
    real_part = trace * 0.5
    sqrt_disc = tl_extra_shim.sqrt(tl_extra_shim.abs(discriminant)) * 0.5

    # Check if discriminant is positive (real eigenvalues) or negative (complex)
    is_real = discriminant >= 0.0

    # For complex eigenvalues, both have the same real part = trace/2
    # For real eigenvalues, they are (trace +/- sqrt_disc)
    e1_real = tl.where(is_real, real_part + sqrt_disc, real_part)
    e1_imag = 0.0

    e2_real = tl.where(is_real, real_part - sqrt_disc, real_part)
    e2_imag = 0.0

    # For complex eigenvalues, we have conjugate pairs
    e1_imag = tl.where(is_real, 0.0, sqrt_disc)
    e2_imag = tl.where(is_real, 0.0, -sqrt_disc)

    return e1_real, e1_imag, e2_real, e2_imag


@triton.jit
def eigvals_2x2_kernel(A, E_real, E_imag, N):
    # Kernel for computing eigenvalues of 2x2 matrices
    # Each program handles one matrix
    pid = tl.program_id(0)
    matrix_offset = pid * 2 * 2

    # Load matrix elements
    a00 = tl.load(A + matrix_offset + 0).to(tl.float32)
    a01 = tl.load(A + matrix_offset + 1).to(tl.float32)
    a10 = tl.load(A + matrix_offset + 2).to(tl.float32)
    a11 = tl.load(A + matrix_offset + 3).to(tl.float32)

    # Compute eigenvalues
    e1_real, e1_imag, e2_real, e2_imag = solve_2x2_eigenvalues(a00, a01, a10, a11)

    # Store results
    output_offset = pid * 2
    tl.store(E_real + output_offset + 0, e1_real)
    tl.store(E_imag + output_offset + 0, e1_imag)
    tl.store(E_real + output_offset + 1, e2_real)
    tl.store(E_imag + output_offset + 1, e2_imag)


def _linalg_eigvals_2x2(A):
    """Compute eigenvalues of 2x2 matrices using Triton kernel."""
    batch_size = 1  # Single matrix

    # Allocate output
    E_real = torch.empty(batch_size * 2, dtype=torch.float32, device=A.device)
    E_imag = torch.empty(batch_size * 2, dtype=torch.float32, device=A.device)

    # Flatten matrix to row-major
    A_flat = A.flatten()

    # Launch kernel - pass tensor directly
    grid = (batch_size,)
    eigvals_2x2_kernel[grid](
        A_flat,
        E_real,
        E_imag,
        batch_size,
    )

    # Combine real and imaginary parts
    E = torch.complex(E_real, E_imag)
    return E


def _linalg_eigvals(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the eigenvalues of a square matrix.

    Args:
        A: A square matrix (2D tensor)

    Returns:
        A complex tensor containing the eigenvalues.
    """
    logger.debug("GEMS _linalg_eigvals")

    # Check input shape
    if A.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {A.dim()}D")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got {A.shape}")

    n = A.shape[0]

    # For 2x2 matrices, use custom Triton kernel
    if n == 2:
        return _linalg_eigvals_2x2(A)
    else:
        # For larger matrices, use torch's implementation
        return torch.ops.aten._linalg_eigvals(A)