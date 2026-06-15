import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def min_max_kernel_1(inp, min_out, max_out, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    # Handle different dtypes
    dtype = inp.type.element_ty
    if tl.constexpr(dtype == tl.float16):
        cdtype = tl.float32
    elif tl.constexpr(dtype == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = dtype

    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0).to(cdtype)
    min_val = tl.min(inp_val)
    max_val = tl.max(inp_val)
    tl.store(min_out + pid, min_val)
    tl.store(max_out + pid, max_val)


@libentry()
@triton.jit
def min_max_kernel_2(mid_min, mid_max, out_min, out_max, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs_min = mid_min + offset
    mid_ptrs_max = mid_max + offset
    mask = offset < mid_size

    dtype = mid_min.type.element_ty
    if tl.constexpr(dtype == tl.float16):
        cdtype = tl.float32
    elif tl.constexpr(dtype == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = dtype

    min_vals = tl.load(mid_ptrs_min, mask=mask, other=0.0).to(cdtype)
    max_vals = tl.load(mid_ptrs_max, mask=mask, other=0.0).to(cdtype)

    min_val = tl.min(min_vals)
    max_val = tl.max(max_vals)

    tl.store(out_min, min_val)
    tl.store(out_max, max_val)


@libentry()
@triton.jit
def quantize_kernel(inp, out, scale, zero_point, N, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N

    dtype = inp.type.element_ty
    if tl.constexpr(dtype == tl.float16):
        cdtype = tl.float32
    elif tl.constexpr(dtype == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = dtype

    inp_val = tl.load(inp + offset, mask=mask, other=0.0).to(cdtype)
    scale_val = tl.load(scale)
    zp_val = tl.load(zero_point)

    # Quantize: round(input / scale) + zero_point
    # Using floor(x + 0.5) for simple rounding
    quantized = tl.floor(inp_val / scale_val + 0.5) + zp_val
    # Clamp to int8 range [-128, 127]
    quantized = tl.clamp(quantized, -128.0, 127.0)

    tl.store(out + offset, quantized.to(tl.int8), mask=mask)


def dynamic_quantize_lstm(input: torch.Tensor) -> tuple:
    """
    Dynamic quantization for LSTM inputs.

    Takes a float tensor and performs dynamic quantization:
    1. Computes min and max values across the entire tensor
    2. Computes scale and zero_point based on the range
    3. Quantizes the tensor using: round(input / scale) + zero_point

    Args:
        input: Input float tensor

    Returns:
        Tuple of (quantized_tensor, scale, zero_point)
        - quantized_tensor: int8 tensor with quantized values
        - scale: float32 scale factor used for quantization
        - zero_point: int8 zero point for quantization
    """
    logger.debug("GEMS DYNAMIC_QUANTIZE_LSTM")

    M = input.numel()
    if M == 0:
        # Empty tensor edge case
        return (torch.empty_like(input, dtype=torch.int8), torch.tensor(1.0), torch.tensor(0, dtype=torch.int8))

    # Ensure contiguous for reduction
    inp = input.contiguous()

    # Determine block sizes
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    # Determine output dtype based on input
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        mid_dtype = torch.float32
    else:
        mid_dtype = inp.dtype

    # Allocate intermediate storage
    mid_min = torch.empty((mid_size,), dtype=mid_dtype, device=inp.device)
    mid_max = torch.empty((mid_size,), dtype=mid_dtype, device=inp.device)
    out_min = torch.empty([], dtype=mid_dtype, device=inp.device)
    out_max = torch.empty([], dtype=mid_dtype, device=inp.device)

    # Kernel 1: Compute partial min/max
    with torch_device_fn.device(inp.device):
        min_max_kernel_1[(mid_size,)](inp, mid_min, mid_max, M, block_size)

    # Kernel 2: Combine partial min/max
    with torch_device_fn.device(inp.device):
        min_max_kernel_2[(1,)](mid_min, mid_max, out_min, out_max, mid_size, block_mid)

    min_val = out_min.item()
    max_val = out_max.item()

    # Compute scale and zero_point for int8 quantization
    # For int8: range is [-128, 127], so 255 discrete values
    # PyTorch's quantize_per_tensor_dynamic uses:
    # scale = (max - min) / 255
    # zero_point = round(-min / scale) - 128
    if min_val == max_val:
        # Edge case: all values are the same
        scale = torch.tensor(1.0, dtype=torch.float32, device=inp.device)
        zero_point = torch.tensor(-128, dtype=torch.int8, device=inp.device)
    else:
        scale_val = (max_val - min_val) / 255.0
        zero_point_val = round(-min_val / scale_val) - 128
        # Clamp zero_point to int8 range
        zero_point_val = max(-128, min(127, zero_point_val))

        scale = torch.tensor(scale_val, dtype=torch.float32, device=inp.device)
        zero_point = torch.tensor(zero_point_val, dtype=torch.int8, device=inp.device)

    # Allocate output tensor
    output = torch.empty_like(inp, dtype=torch.int8)

    # Kernel 3: Quantize the tensor
    BLOCK_N = 1024
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        quantize_kernel[grid](inp, output, scale, zero_point, M, BLOCK_N=BLOCK_N)

    return (output, scale, zero_point)