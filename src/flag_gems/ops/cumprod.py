import functools
import logging
import math

import torch
import triton
import triton.language as tl
from torch._prims_common import is_boolean_dtype, is_integer_dtype

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import get_device_properties, libentry
from flag_gems.utils import triton_lang_extension as tle

device = device.name
logger = logging.getLogger(__name__)


@functools.lru_cache
def get_num_sms(idx: int) -> int:
    return get_device_properties(idx).multi_processor_count


@tl.constexpr
def get_scan_accum_type(inp_dtype: tl.dtype) -> tl.dtype:
    if inp_dtype.is_bf16() or inp_dtype.is_fp16():
        return tl.float32
    if inp_dtype.is_int():  # signed or not(including bool)
        return tl.int64
    else:
        return inp_dtype


@triton.jit
def tl_log_prod(x):
    """Compute log of product: log(prod(x)) = sum(log(x))"""
    # For numerical stability with very small values, clamp log inputs
    abs_x = tl.abs(x)
    # Avoid log(0) by using a small threshold
    abs_x = tl.where(abs_x == 0, 1e-40, abs_x)
    return tl.sum(tl.log(abs_x))


@triton.jit
def tl_sign_prod(x):
    """Compute product of signs: prod(sign(x))"""
    # sign(x) = 1 if x >= 0, -1 if x < 0
    # product of signs = (-1)^(number of negative values)
    neg_count = tl.sum(tl.where(x < 0, 1, 0))
    # If neg_count is odd, product is -1; otherwise 1
    return tl.where(neg_count % 2 == 0, 1.0, -1.0)


@triton.jit
def tl_prod_vals(x):
    """Compute product of a 1D tensor using log/exp for magnitude and sign for sign"""
    log_sum = tl_log_prod(x)
    prod_mag = tl.exp(log_sum)
    sign_prod = tl_sign_prod(x)
    return prod_mag * sign_prod


@libentry()
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def scan_part_prod_kernel(
    inp,
    out,
    partial_prod,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl.cumprod(inp_vals, axis=0)

    part_prod_via_prod = tl_prod_vals(inp_vals)

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    partial_prod_ptrs = partial_prod + pid
    tl.store(partial_prod_ptrs, part_prod_via_prod)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def add_base_prod_kernel(
    out,
    partial_prod,
    n_elements,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid > 0:
        partial_prod_ptrs = partial_prod + pid - 1
        last_part_prod_via_prod = tl.load(partial_prod_ptrs)

        final_vals = out_vals * last_part_prod_via_prod
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def scan_part_prod_abc_kernel(
    inp,
    out,
    partial_prod,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tle.program_id(0)
    pid_b = tle.program_id(1)
    pid_c = tle.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    part_offset = base_part_offset + pid_b * C

    mask = b_idx < B
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl.cumprod(inp_vals, axis=0)

    part_prod_via_prod = tl_prod_vals(inp_vals)

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    partial_prod_ptrs = partial_prod + part_offset
    tl.store(partial_prod_ptrs, part_prod_via_prod)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def add_base_prod_abc_kernel(
    out,
    partial_prod,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tle.program_id(0)
    pid_b = tle.program_id(1)
    pid_c = tle.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    base_offset = a_idx * B * C + c_idx
    offset = base_offset + b_idx * C
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C

    mask = b_idx < B
    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid_b > 0:
        partial_prod_ptrs = partial_prod + last_part_offset
        last_part_prod_via_prod = tl.load(partial_prod_ptrs)

        final_vals = out_vals * last_part_prod_via_prod
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


def scan_then_fan_col(inp, out, n_ele, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if n_ele <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_prod = torch.empty(part_num, dtype=dtype, device=inp.device)

    grid = (part_num,)
    with torch_device_fn.device(inp.device):
        scan_part_prod_kernel[grid](inp, out, partial_prod, n_ele, part_num, BLOCK_SIZE)

    if part_num >= 2:
        scan_then_fan_col(partial_prod, partial_prod, part_num, dtype)
        with torch_device_fn.device(inp.device):
            add_base_prod_kernel[grid](out, partial_prod, n_ele, part_num, BLOCK_SIZE)


def scan_then_fan(inp, out, A, B, C, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_prod = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)

    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_prod_abc_kernel[grid](
            inp, out, partial_prod, B, C, part_num, BLOCK_SIZE
        )

    if part_num >= 2:
        scan_then_fan(partial_prod, partial_prod, A, part_num, C, dtype)
        with torch_device_fn.device(inp.device):
            add_base_prod_abc_kernel[grid](out, partial_prod, B, C, part_num, BLOCK_SIZE)


def cumprod_wrapper(inp, dim=1, dtype=None, out=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
            dtype = torch.int64
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    compute_dtype = out.dtype
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if K == 1:  # row scan
        reduce_then_scan_row(inp, out, M, N, compute_dtype)
    else:  # col scan
        scan_then_fan(inp, out, M, N, K, compute_dtype)

    return out


def reduce_then_scan_row(x, out, M, N, compute_dtype):
    if N <= 16384:  # persistent
        TILE_SIZE = triton.next_power_of_2(N)
        num_warps = 8 if TILE_SIZE > 2048 else 4
        reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
            x, out, N, TILE_SIZE, num_warps=num_warps
        )
        return out

    TILE_SIZE = min(4096, triton.next_power_of_2(N))
    num_warps = 8 if TILE_SIZE > 2048 else 4
    num_tiles = triton.cdiv(N, TILE_SIZE)
    max_ctas = get_num_sms(x.device.index) * 4
    num_ctas = min(num_tiles, max_ctas)
    ROOT_SCAN_TILE_SIZE = triton.next_power_of_2(num_ctas)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    block_prods = torch.empty(
        (
            M,
            num_ctas,
        ),
        dtype=compute_dtype,
        device=x.device,
    )
    block_inclusive_prefix = torch.empty(
        (
            M,
            num_ctas,
        ),
        dtype=compute_dtype,
        device=x.device,
    )

    # 3-kernel implementation
    reduce_then_scan_block_prod_kernel_row[(M, num_ctas, 1, 1)](
        x, block_prods, N, tiles_per_cta, TILE_SIZE, num_warps=num_warps
    )
    reduce_then_scan_root_scan_kernel_row[(M, 1, 1)](
        block_prods,
        block_inclusive_prefix,
        num_ctas,
        ROOT_SCAN_TILE_SIZE,
        num_warps=num_warps,
    )
    reduce_then_scan_block_scan_kernel_row[(M, num_ctas, 1)](
        x,
        block_inclusive_prefix,
        out,
        N,
        num_ctas,
        tiles_per_cta,
        TILE_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def reduce_then_scan_block_prod_kernel_row(
    in_ptr,
    block_prod_ptr,
    N,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    """The same kernel as the block product in parallel reduce"""
    pid_n = tl.program_id(1).to(tl.int64)
    pid_m = tl.program_id(0).to(tl.int64)
    num_programs_n = tl.num_programs(1)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)

    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)
    # Initialize with identity element for multiplication (1)
    acc = tl.full((TILE_SIZE,), 1.0, dtype=acc_dtype)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        x = tl.load(in_ptr + pid_m * N + offsets, mask=offsets < N).to(acc_dtype)
        acc = acc * x
    block_prod = tl_prod_vals(acc)
    tl.store(
        block_prod_ptr + pid_m * num_programs_n + pid_n, block_prod, cache_modifier=".cg"
    )


@triton.jit
def reduce_then_scan_root_scan_kernel_row(
    in_ptr, out_ptr, N, TILE_SIZE: tl.constexpr
):
    """Almost The same kernel as the persistent scan kernel"""
    pid = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TILE_SIZE)
    mask = offsets < N
    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)
    # Load values - use 1 as identity for multiplication
    x = tl.load(in_ptr + pid * N + offsets, mask=mask, other=1).to(acc_dtype)
    out = tl.cumprod(x, 0)
    tl.store(out_ptr + pid * N + offsets, out, mask=mask)


@triton.jit
def reduce_then_scan_block_scan_kernel_row(
    in_ptr,
    previous_prod_ptr,
    out_ptr,
    N,
    num_tiles_n,
    tiles_per_cta,
    TILE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    block_offset = pid_n * (tiles_per_cta * TILE_SIZE)
    block_end = min(block_offset + tiles_per_cta * TILE_SIZE, N)
    acc_dtype: tl.constexpr = get_scan_accum_type(in_ptr.type.element_ty)

    # Load prefix - use 1 as identity for multiplication
    prefix = tl.load(
        previous_prod_ptr + pid_m * num_tiles_n + pid_n - 1, mask=pid_n > 0, other=1
    ).to(acc_dtype)
    for start in range(block_offset, block_end, TILE_SIZE):
        offsets = start + tl.arange(0, TILE_SIZE)
        mask = offsets < N
        x = tl.load(in_ptr + pid_m * N + offsets, mask=mask).to(acc_dtype)
        tile_scan = prefix * tl.cumprod(x, 0)
        # Compute block product for prefix update
        block_prod = tl_prod_vals(x)
        prefix = prefix * block_prod
        tl.store(
            out_ptr + pid_m * N + offsets, tile_scan, mask=mask, cache_modifier=".cg"
        )


def cumprod(inp, dim=1, *, dtype=None):
    logger.debug("GEMS cumprod")
    return cumprod_wrapper(inp, dim, dtype)


def cumprod_out(inp, dim=1, *, dtype=None, out):
    logger.debug("GEMS cumprod_OUT")
    return cumprod_wrapper(inp, dim, dtype, out)