import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmv_kernel_(
    A,
    B,
    Inp,
    N,
    M,
    alpha,
    beta,
    stride_an,
    stride_am,
    stride_bm,
    stride_in,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tle.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)[:, None]
    Inp_ptrs = Inp + offset_n * stride_in
    inp = tl.load(Inp_ptrs, mask=n_mask, other=0.0).to(tl.float32)
    out_block = acc * alpha + inp * beta
    tl.store(Inp_ptrs, out_block, mask=n_mask)


def addmv_(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS ADDMV_")
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert broadcastable_to(self.shape, (mat.shape[0],)), "Incompatible self shape"
    N, M = mat.shape
    self = self.broadcast_to((N,))
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(mat.device):
        addmv_kernel_[grid](
            mat,
            vec,
            self,
            N,
            M,
            alpha,
            beta,
            mat.stride(0),
            mat.stride(1),
            vec.stride(0),
            self.stride(0),
        )
    return self