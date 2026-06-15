import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def linalg_det_kernel_2x2(A, out, stride_ab, stride_am, stride_an, stride_ob):
    """Compute determinant for 2x2 matrix"""
    pid_b = tle.program_id(0)
    A_ptr = A + pid_b * stride_ab
    out_ptr = out + pid_b * stride_ob

    # Load matrix elements
    a00 = tl.load(A_ptr + 0 * stride_am + 0 * stride_an).to(tl.float32)
    a01 = tl.load(A_ptr + 0 * stride_am + 1 * stride_an).to(tl.float32)
    a10 = tl.load(A_ptr + 1 * stride_am + 0 * stride_an).to(tl.float32)
    a11 = tl.load(A_ptr + 1 * stride_am + 1 * stride_an).to(tl.float32)

    # det = a00 * a11 - a01 * a10
    det = a00 * a11 - a01 * a10
    tl.store(out_ptr, det.to(out.dtype.element_ty))


@libentry()
@triton.jit
def linalg_det_kernel_3x3(A, out, stride_ab, stride_am, stride_an, stride_ob):
    """Compute determinant for 3x3 matrix using rule of Sarrus"""
    pid_b = tle.program_id(0)
    A_ptr = A + pid_b * stride_ab
    out_ptr = out + pid_b * stride_ob

    # Load matrix elements
    a00 = tl.load(A_ptr + 0 * stride_am + 0 * stride_an).to(tl.float32)
    a01 = tl.load(A_ptr + 0 * stride_am + 1 * stride_an).to(tl.float32)
    a02 = tl.load(A_ptr + 0 * stride_am + 2 * stride_an).to(tl.float32)
    a10 = tl.load(A_ptr + 1 * stride_am + 0 * stride_an).to(tl.float32)
    a11 = tl.load(A_ptr + 1 * stride_am + 1 * stride_an).to(tl.float32)
    a12 = tl.load(A_ptr + 1 * stride_am + 2 * stride_an).to(tl.float32)
    a20 = tl.load(A_ptr + 2 * stride_am + 0 * stride_an).to(tl.float32)
    a21 = tl.load(A_ptr + 2 * stride_am + 1 * stride_an).to(tl.float32)
    a22 = tl.load(A_ptr + 2 * stride_am + 2 * stride_an).to(tl.float32)

    # Rule of Sarrus:
    # det = a00*a11*a22 + a01*a12*a20 + a02*a10*a21 - a02*a11*a20 - a00*a12*a21 - a01*a10*a22
    det = (a00 * a11 * a22 + a01 * a12 * a20 + a02 * a10 * a21
           - a02 * a11 * a20 - a00 * a12 * a21 - a01 * a10 * a22)
    tl.store(out_ptr, det.to(out.dtype.element_ty))


@libentry()
@triton.jit
def linalg_det_kernel_4x4(A, out, stride_ab, stride_am, stride_an, stride_ob):
    """Compute determinant for 4x4 matrix using Laplace expansion"""
    pid_b = tle.program_id(0)
    A_ptr = A + pid_b * stride_ab
    out_ptr = out + pid_b * stride_ob

    # Load matrix elements
    a00 = tl.load(A_ptr + 0 * stride_am + 0 * stride_an).to(tl.float32)
    a01 = tl.load(A_ptr + 0 * stride_am + 1 * stride_an).to(tl.float32)
    a02 = tl.load(A_ptr + 0 * stride_am + 2 * stride_an).to(tl.float32)
    a03 = tl.load(A_ptr + 0 * stride_am + 3 * stride_an).to(tl.float32)
    a10 = tl.load(A_ptr + 1 * stride_am + 0 * stride_an).to(tl.float32)
    a11 = tl.load(A_ptr + 1 * stride_am + 1 * stride_an).to(tl.float32)
    a12 = tl.load(A_ptr + 1 * stride_am + 2 * stride_an).to(tl.float32)
    a13 = tl.load(A_ptr + 1 * stride_am + 3 * stride_an).to(tl.float32)
    a20 = tl.load(A_ptr + 2 * stride_am + 0 * stride_an).to(tl.float32)
    a21 = tl.load(A_ptr + 2 * stride_am + 1 * stride_an).to(tl.float32)
    a22 = tl.load(A_ptr + 2 * stride_am + 2 * stride_an).to(tl.float32)
    a23 = tl.load(A_ptr + 2 * stride_am + 3 * stride_an).to(tl.float32)
    a30 = tl.load(A_ptr + 3 * stride_am + 0 * stride_an).to(tl.float32)
    a31 = tl.load(A_ptr + 3 * stride_am + 1 * stride_an).to(tl.float32)
    a32 = tl.load(A_ptr + 3 * stride_am + 2 * stride_an).to(tl.float32)
    a33 = tl.load(A_ptr + 3 * stride_am + 3 * stride_an).to(tl.float32)

    # Compute 3x3 minors
    m00 = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
    m01 = a10 * (a22 * a33 - a23 * a32) - a12 * (a20 * a33 - a23 * a30) + a13 * (a20 * a32 - a22 * a30)
    m02 = a10 * (a21 * a33 - a23 * a31) - a11 * (a20 * a33 - a23 * a30) + a13 * (a20 * a31 - a21 * a30)
    m03 = a10 * (a21 * a32 - a22 * a31) - a11 * (a20 * a32 - a22 * a30) + a12 * (a20 * a31 - a21 * a30)

    # Laplace expansion along first row
    det = a00 * m00 - a01 * m01 + a02 * m02 - a03 * m03
    tl.store(out_ptr, det.to(out.dtype.element_ty))


@libentry()
@triton.jit
def linalg_det_kernel_n(
    A,
    out,
    n,
    batch_size,
    stride_ab,
    stride_am,
    stride_an,
    stride_ob,
):
    """Compute determinant for n x n matrix (n > 4) using Gaussian elimination without pivoting"""
    pid_b = tle.program_id(0)
    batch_idx = pid_b

    A_ptr = A + batch_idx * stride_ab
    out_ptr = out + batch_idx * stride_ob

    # Using simple Gaussian elimination
    det = tl.cast(1.0, tl.float32)

    # For each column
    for i in range(n):
        # Load pivot
        ptr_ii = A_ptr + i * stride_am + i * stride_an
        pivot = tl.load(ptr_ii).to(tl.float32)

        # Multiply determinant by pivot
        det = det * pivot

        # Eliminate below
        if i < n - 1:
            for k in range(i + 1, n):
                ptr_ki = A_ptr + k * stride_am + i * stride_an
                factor = tl.load(ptr_ki).to(tl.float32)
                if i > 0:
                    factor = factor / tl.load(A_ptr + i * stride_am + i * stride_an).to(tl.float32)
                if tl.abs(factor) > 1e-10:
                    for j in range(i + 1, n):
                        ptr_kj = A_ptr + k * stride_am + j * stride_an
                        ptr_ij = A_ptr + i * stride_am + j * stride_an
                        val_kj = tl.load(ptr_kj).to(tl.float32)
                        val_ij = tl.load(ptr_ij).to(tl.float32)
                        new_val = val_kj - factor * val_ij
                        tl.store(ptr_kj, new_val)

    tl.store(out_ptr, det.to(out.dtype.element_ty))


def linalg_det(A):
    logger.debug("GEMS linalg_det")

    # Ensure input is contiguous and get shape
    A = A.contiguous()
    shape = A.shape

    # Input must be (*, n, n) where n is the last two dimensions
    if len(shape) < 2:
        raise ValueError("Input must have at least 2 dimensions")

    n = shape[-1]
    m = shape[-2]

    if n != m:
        raise ValueError("Input must be square matrices")

    # Batch dimensions
    batch_shape = shape[:-2]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # Output shape is (*)
    output_shape = batch_shape if batch_shape else ()
    out = torch.empty(output_shape, dtype=A.dtype, device=A.device)

    # Handle empty batch
    if batch_size == 0:
        return out

    stride_am = A.stride(-2)
    stride_an = A.stride(-1)

    # Handle single matrix vs batch
    if batch_size == 1:
        # Single matrix case
        if n == 1:
            # 1x1 matrix: just return the single element
            out = A.squeeze(-1).squeeze(-1)
            return out
        elif n == 2:
            grid = (1, 1, 1)
            linalg_det_kernel_2x2[grid](A, out, 0, stride_am, stride_an, 0)
        elif n == 3:
            grid = (1, 1, 1)
            linalg_det_kernel_3x3[grid](A, out, 0, stride_am, stride_an, 0)
        elif n == 4:
            grid = (1, 1, 1)
            linalg_det_kernel_4x4[grid](A, out, 0, stride_am, stride_an, 0)
        else:
            grid = (1,)
            linalg_det_kernel_n[grid](
                A, out, n, batch_size, 0, stride_am, stride_an, 0
            )
    else:
        # Batched case
        stride_ab = A.stride(-3)
        stride_ob = out.stride(-1) if out.ndim > 0 else 0

        if n == 2:
            grid = (batch_size, 1, 1)
            linalg_det_kernel_2x2[grid](A, out, stride_ab, stride_am, stride_an, stride_ob)
        elif n == 3:
            grid = (batch_size, 1, 1)
            linalg_det_kernel_3x3[grid](A, out, stride_ab, stride_am, stride_an, stride_ob)
        elif n == 4:
            grid = (batch_size, 1, 1)
            linalg_det_kernel_4x4[grid](A, out, stride_ab, stride_am, stride_an, stride_ob)
        else:
            grid = (batch_size,)
            linalg_det_kernel_n[grid](
                A, out, n, batch_size, stride_ab, stride_am, stride_an, stride_ob
            )

    return out