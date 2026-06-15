import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def l2norm_kernel_1(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(x * x)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def l2norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


def L2Norm(A, ord=2):
    """Compute L2 norm (Euclidean norm) of a tensor.

    L2Norm computes sqrt(sum(x^2)) which is the L2 norm (also known as
    Euclidean norm or Frobenius norm for matrices).

    Args:
        A: Input tensor
        ord: Order of the norm (default 2). Currently only ord=2 is supported.
             This parameter is accepted for compatibility but ignored.
    """
    logger.debug("GEMS L2NORM")
    A = A.contiguous()
    M = A.numel()

    dtype = A.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"L2Norm not implemented for {dtype}")

    BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
    BLOCK_MID = triton.next_power_of_2(MID_SIZE)

    mid = torch.empty([MID_SIZE], dtype=torch.float32, device=A.device)
    out = torch.empty([], dtype=torch.float32, device=A.device)

    with torch_device_fn.device(A.device):
        l2norm_kernel_1[(MID_SIZE,)](A, mid, M, BLOCK_SIZE)
        l2norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)

    return out.to(dtype)


def L2Norm_(A):
    """In-place L2 norm (computes norm in-place, stores result in A).

    Note: This is not a true in-place operation as L2 norm reduces
    all elements to a single scalar value. This function computes
    the L2 norm and stores the result in the input tensor.
    """
    logger.debug("GEMS L2NORM_")
    result = L2Norm(A)
    A.copy_(result)
    return A