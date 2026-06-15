import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def aminmax_kernel_1(
    inp,
    mid_min,
    mid_max,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    amin_val = tl.min(inp_val)
    amax_val = tl.max(inp_val)
    mid_min_ptr = mid_min + pid
    mid_max_ptr = mid_max + pid
    tl.store(mid_min_ptr, amin_val)
    tl.store(mid_max_ptr, amax_val)


@libentry()
@triton.jit
def aminmax_kernel_2(mid_min, mid_max, out_min, out_max, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_min_ptrs = mid_min + offset
    mid_max_ptrs = mid_max + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid_min.type.element_ty)
    max_value = get_dtype_max(mid_max.type.element_ty)
    mid_min_val = tl.load(mid_min_ptrs, mask=mask, other=min_value)
    mid_max_val = tl.load(mid_max_ptrs, mask=mask, other=max_value)
    amin_val = tl.min(mid_min_val)
    amax_val = tl.max(mid_max_val)
    tl.store(out_min, amin_val)
    tl.store(out_max, amax_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def aminmax_kernel(
    inp,
    out_min,
    out_max,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty

    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out_min = out_min + rows
    out_max = out_max + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    inf_value = float("inf")
    neg_inf_value = float("-inf")
    _amin = tl.full([BLOCK_M, BLOCK_N], inf_value, dtype=acc_type)
    _amax = tl.full([BLOCK_M, BLOCK_N], neg_inf_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        # Load twice with different fill values for min and max
        a_for_amin = tl.load(inp + cols, mask, other=inf_value).to(acc_type)
        a_for_amax = tl.load(inp + cols, mask, other=neg_inf_value).to(acc_type)
        _amin = tl.minimum(_amin, a_for_amin)
        _amax = tl.maximum(_amax, a_for_amax)
    amin = tl.min(_amin, axis=1)[:, None]
    amax = tl.max(_amax, axis=1)[:, None]
    tl.store(out_min, amin, row_mask)
    tl.store(out_max, amax, row_mask)


def aminmax(inp, *, dim=None, keepdim=False):
    logger.debug("GEMS AMINMAX")
    if isinstance(dim, int):
        dim = [dim]
    if dim is None or len(dim) == 0:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype

        mid_min = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_max = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        if not keepdim:
            out_min = torch.empty([], dtype=dtype, device=inp.device)
            out_max = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            out_min = torch.empty(shape, dtype=dtype, device=inp.device)
            out_max = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            aminmax_kernel_1[(mid_size, 1)](
                inp,
                mid_min,
                mid_max,
                M,
                block_size,
            )
            aminmax_kernel_2[(1, 1)](
                mid_min, mid_max, out_min, out_max, mid_size, block_mid
            )
        return out_min, out_max
    else:
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out_min = torch.empty(shape, dtype=dtype, device=inp.device)
        out_max = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            aminmax_kernel[grid](inp, out_min, out_max, M, N)
        if not keepdim:
            out_min = out_min.squeeze(dim=dim)
            out_max = out_max.squeeze(dim=dim)
        return out_min, out_max