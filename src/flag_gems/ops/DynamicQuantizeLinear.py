import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# ============================================================================
# Quantization kernel
# ============================================================================

@libentry()
@triton.jit
def quantize_kernel(
    input_ptr,
    output_ptr,
    scale,
    zero_point,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N

    x = tl.load(input_ptr + offset, mask=mask, other=0.0)

    # Load scale and zero_point as scalars
    scale_val = tl.load(scale)
    zp_val = tl.load(zero_point)

    # Quantization: round(x / scale) + zero_point, clamped to [-128, 127]
    quantized = tl.math.div_rn(x, scale_val) + zp_val
    quantized = tl.clamp(quantized, -128.0, 127.0)

    tl.store(output_ptr + offset, quantized, mask=mask)


def quantize(inp, scale, zero_point):
    """Quantize the input tensor using given scale and zero_point."""
    N = inp.numel()
    if N == 0:
        return torch.empty((0,), dtype=torch.int8, device=inp.device)

    inp = inp.contiguous()

    # Output as int8
    output = torch.empty_like(inp, dtype=torch.int8)

    block_size = triton.next_power_of_2(triton.cdiv(N, 256))
    if block_size < 32:
        block_size = 32

    with torch_device_fn.device(inp.device):
        grid = (triton.cdiv(N, block_size),)
        quantize_kernel[grid](
            inp, output, scale, zero_point, N, block_size
        )

    return output


def DynamicQuantizeLinear(A):
    """
    Dynamic quantization for linear layers.

    Computes scale and zero_point dynamically from input tensor,
    then quantizes the tensor to int8.

    Args:
        A: Input tensor (float32, float16, bfloat16)

    Returns:
        tuple: (output, scale, zero_point)
            - output: Quantized tensor (int8)
            - scale: Quantization scale (float32)
            - zero_point: Quantization zero point (int8)
    """
    logger.debug("GEMS DynamicQuantizeLinear")

    # Handle empty tensor
    if A.numel() == 0:
        output = torch.empty((0,), dtype=torch.int8, device=A.device)
        scale = torch.tensor(0.0, dtype=torch.float32, device=A.device)
        zero_point = torch.tensor(0, dtype=torch.int8, device=A.device)
        return output, scale, zero_point

    # Compute min and max values using torch (more reliable)
    min_val_cpu = A.min().item()
    max_val_cpu = A.max().item()

    # Compute scale and zero_point
    # Qmin, Qmax for int8 = -128, 127
    qmin, qmax = -128, 127
    qrange = qmax - qmin  # 255

    if max_val_cpu == min_val_cpu:
        # All values are the same
        abs_val = abs(min_val_cpu)
        if abs_val > 0:
            # For symmetric quantization: scale = |value| / 255
            scale_val = abs_val / qrange
            # zero_point: -128 for positive, 127 for negative
            zero_point_val = -128 if min_val_cpu >= 0 else 127
        else:
            # For zero value, use arbitrary scale and zero_point
            scale_val = 0.1
            zero_point_val = 127
    else:
        # Use standard asymmetric quantization formula
        scale_val = (max_val_cpu - min_val_cpu) / qrange
        zero_point_val = qmin - round(min_val_cpu / scale_val)

    scale = torch.tensor(scale_val, dtype=torch.float32, device=A.device)
    zero_point = torch.tensor(zero_point_val, dtype=torch.int8, device=A.device)

    # Quantize using Triton kernel
    output = quantize(A, scale, zero_point)

    return output, scale, zero_point