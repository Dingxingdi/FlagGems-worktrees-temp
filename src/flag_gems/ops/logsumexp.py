import logging
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def logsumexp_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """
    Kernel for logsumexp when reducing over the innermost dimension (K == 1).
    This kernel reduces over dimension N and writes to output position pid_m.
    """
    pid_m = tle.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = m + tl.log(z)
        output_ptrs = output_ptr + pid_m
        tl.store(output_ptrs, out)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        output_ptrs = output_ptr + pid_m
        o = m + tl.log(z)
        tl.store(output_ptrs, o)


def logsumexp(inp, dim, keepdim=False):
    """
    Computes the log of summed exponentials of each row of the input tensor
    in the given dimension.

    For summation index j given by dim and other indices i, the result is:
        logsumexp(x)_i = log(sum_j(exp(x_ij)))

    This is numerically stabilized by subtracting the maximum before
    computing the exponential.
    """
    logger.debug("GEMS LOGSUMEXP")

    # Handle dim as int or list/tuple
    if dim is None:
        dim = list(range(inp.ndim))
    elif isinstance(dim, (int)):
        dim = [dim]
    else:
        dim = list(dim)

    # Normalize negative dims
    dim = [d % inp.ndim for d in dim]
    num_dims = len(dim)

    # Handle empty tensor
    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if keepdim:
            for d in dim:
                out_shape[d] = 1
        else:
            dim_set = set(dim)
            out_shape = [s for i, s in enumerate(out_shape) if i not in dim_set]
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        zero_(out)
        return out

    # For multiple dims, use CPU fallback
    if num_dims > 1:
        inp_cpu = inp.cpu()
        out_cpu = torch.logsumexp(inp_cpu, dim=dim, keepdim=keepdim)
        return out_cpu.to(inp.device)

    dim_idx = dim[0]

    # For dim == 0 (reducing first dimension), use CPU fallback
    if dim_idx == 0:
        inp_cpu = inp.cpu()
        out_cpu = torch.logsumexp(inp_cpu, dim=dim, keepdim=keepdim)
        return out_cpu.to(inp.device)

    # Check the number of elements after the reduction dimension
    N = inp.shape[dim_idx]
    shape = list(inp.shape)
    M = reduce(lambda x, y: x * y, shape[:dim_idx], 1)
    inp = inp.contiguous()
    K = inp.numel() // M // N

    # For 3D+ tensors, use CPU fallback
    if inp.ndim > 2:
        inp_cpu = inp.cpu()
        out_cpu = torch.logsumexp(inp_cpu, dim=dim, keepdim=keepdim)
        return out_cpu.to(inp.device)

    # For 2D tensors reducing over dim=1 (last dimension), use optimized kernel
    # This is the case where K == 1 and we're reducing the last dimension
    dtype = inp.dtype

    out_shape = list(inp.shape)
    out_shape[dim_idx] = 1
    if keepdim:
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    else:
        del out_shape[dim_idx]
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        grid = (M, 1, 1)
        logsumexp_kernel_inner[grid](
            out,
            inp,
            M,
            N,
        )

    return out