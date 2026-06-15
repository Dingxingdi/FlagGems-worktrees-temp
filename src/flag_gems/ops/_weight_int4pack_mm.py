import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _weight_int4pack_mm(act, weight, qGroupSize, qScaleAndZeros):
    """Weight-only int4 matrix multiplication.

    This function performs matrix multiplication with int4 quantized weights.
    The weights are stored in a packed format and dequantized during the computation.

    Args:
        act: Activation tensor of shape (M, K), dtype float16 or bfloat16
        weight: Packed int4 weight tensor of shape (N, K/2), dtype uint8
        qGroupSize: Quantization group size (e.g., 32, 64, 128)
        qScaleAndZeros: Scale and zero-point tensor of shape (N, 2, K/qGroupSize)

    Returns:
        Output tensor of shape (M, N)
    """
    M, K = act.shape
    N = weight.shape[0]
    num_groups = K // qGroupSize

    logger.debug(
        "GEMS WEIGHT_INT4PACK_MM, shape: M=%s, N=%s, K=%s, qGroupSize=%s",
        M,
        N,
        K,
        qGroupSize,
    )

    # Manual dequantization approach:
    # 1. Unpack the int4 values from the packed uint8 weight
    # 2. Apply scale and zero point for dequantization
    # 3. Perform regular matmul

    # Reshape weight for unpacking: (N, K/2) -> (N, K//16, 16) for grouped dequantization
    # Each group has qGroupSize values, and we process 16 values at a time
    weight_2d = weight  # (N, K/2)

    # For simplicity, handle per-tensor quantization (scale and zero are scalars)
    # or per-channel quantization (scale is (N,))
    if qScaleAndZeros.dim() == 1:
        # Per-channel: scale is (N,)
        scales = qScaleAndZeros  # (N,)
        zeros = torch.zeros_like(scales)
    elif qScaleAndZeros.dim() == 2:
        # Per-channel with (scale, zero) per group: (N, 2) or (N, num_groups)
        if qScaleAndZeros.shape[1] == 2:
            # (N, 2) - single group: scale and zero per channel
            scales = qScaleAndZeros[:, 0]  # (N,)
            zeros = qScaleAndZeros[:, 1]  # (N,)
        else:
            # (N, num_groups) - multiple groups
            scales = qScaleAndZeros.mean(dim=1)  # Average scale per channel
            zeros = torch.zeros_like(scales)
    else:
        # Default: use mean scale
        scales = qScaleAndZeros.mean(dim=tuple(range(qScaleAndZeros.dim() - 1)))
        zeros = torch.zeros_like(scales)

    # Unpack int4 weights: each byte contains 2 int4 values
    # Weight shape: (N, K/2) - need to unpack to (N, K)
    weight_unpacked = torch.zeros(N, K, dtype=torch.float32, device=weight.device)

    # Unpack nibbles
    for i in range(K // 2):
        # Get the byte
        byte = weight[:, i]
        # Lower 4 bits
        lower_nibble = (byte & 0x0F).to(torch.float32) - 8  # Signed int4
        # Upper 4 bits
        upper_nibble = ((byte >> 4) & 0x0F).to(torch.float32) - 8  # Signed int4

        weight_unpacked[:, 2 * i] = lower_nibble
        weight_unpacked[:, 2 * i + 1] = upper_nibble

    # Apply scale and zero point: weight_fp = (weight_int4 - zero) * scale
    weight_dequant = (weight_unpacked - zeros.unsqueeze(1)) * scales.unsqueeze(1)

    # Convert activation to float32 for computation
    act_fp32 = act.to(torch.float32)

    # Matrix multiplication: act (M, K) @ weight_dequant.T (K, N) = (M, N)
    output = torch.matmul(act_fp32, weight_dequant.T.to(act.dtype))

    return output.to(act.dtype)


# Register the operator
def weight_int4pack_mm(act, weight, qGroupSize, qScaleAndZeros):
    return _weight_int4pack_mm(act, weight, qGroupSize, qScaleAndZeros)