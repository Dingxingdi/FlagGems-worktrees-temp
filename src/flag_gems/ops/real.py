import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _real_kernel(ri_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # 复数被 view_as_real 展平为 [real0, imag0, real1, imag1, ...]
    # 实部在偶数位置
    base = offsets * 2
    real_val = tl.load(ri_ptr + base, mask=mask)
    tl.store(out_ptr + offsets, real_val, mask=mask)


def real(A):
    logger.debug("GEMS REAL")
    if A.is_complex():
        # 使用 view_as_real 转换复数为实数张量
        # 转换后 shape 会翻倍：[N] -> [2N]
        ri = torch.view_as_real(A)

        # 确定输出 dtype：complex64 -> float32, complex128 -> float64
        if A.dtype == torch.complex64:
            out_dtype = torch.float32
        else:
            out_dtype = torch.float64

        # 如果输入是 contiguous 的，直接在原位处理
        if A.is_contiguous():
            out = torch.empty(A.shape, dtype=out_dtype, device=A.device)
            n_elements = A.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            with torch_device_fn.device(A.device):
                _real_kernel[grid](ri, out, n_elements, BLOCK_SIZE=1024)
            return out
        else:
            # 非 contiguous 输入，使用 PyTorch 原生实现
            return torch.ops.aten.real(A)
    else:
        # 对于非复数输入，使用 aten real
        return torch.ops.aten.real(A)