import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _convert_weight_to_int4pack_for_cpu_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    innerKTiles,
    num_output_elements,
    BLOCK_SIZE: tl.constexpr = 128,
):
    """
    Convert uint8 weight tensor to int4 packed format.

    Note: This implementation provides basic int4 packing. The actual
    PyTorch implementation uses a specific quantization format for
    NVIDIA Tensor Core int4 inference.
    """
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    # Calculate start and end for this program
    start_pid = pid
    stride = num_pid

    for off in range(start_pid, num_output_elements, stride * BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_output_elements

        # Calculate input byte index (each output element represents 4 bytes)
        byte_idx = offs * 4

        # Calculate row and column in original tensor
        row_idx = byte_idx // N
        col_idx = byte_idx % N

        # Load 4 bytes for each output element
        byte_offsets = row_idx * N + col_idx

        # Load the 4 bytes
        x0 = tl.load(input_ptr + byte_offsets, mask=mask, other=0)
        x1 = tl.load(input_ptr + byte_offsets + 1, mask=mask, other=0)
        x2 = tl.load(input_ptr + byte_offsets + 2, mask=mask, other=0)
        x3 = tl.load(input_ptr + byte_offsets + 3, mask=mask, other=0)

        # Extract nibbles (4-bit values)
        nibble0_lo = x0 & 0x0F
        nibble0_hi = (x0 >> 4) & 0x0F
        nibble1_lo = x1 & 0x0F
        nibble1_hi = (x1 >> 4) & 0x0F
        nibble2_lo = x2 & 0x0F
        nibble2_hi = (x2 >> 4) & 0x0F
        nibble3_lo = x3 & 0x0F
        nibble3_hi = (x3 >> 4) & 0x0F

        # Convert unsigned int4 (0-15) to signed int4 (-8 to 7)
        i4_0 = tl.where(nibble0_lo >= 8, nibble0_lo - 16, nibble0_lo)
        i4_1 = tl.where(nibble0_hi >= 8, nibble0_hi - 16, nibble0_hi)
        i4_2 = tl.where(nibble1_lo >= 8, nibble1_lo - 16, nibble1_lo)
        i4_3 = tl.where(nibble1_hi >= 8, nibble1_hi - 16, nibble1_hi)
        i4_4 = tl.where(nibble2_lo >= 8, nibble2_lo - 16, nibble2_lo)
        i4_5 = tl.where(nibble2_hi >= 8, nibble2_hi - 16, nibble2_hi)
        i4_6 = tl.where(nibble3_lo >= 8, nibble3_lo - 16, nibble3_lo)
        i4_7 = tl.where(nibble3_hi >= 8, nibble3_hi - 16, nibble3_hi)

        # Cast to int32 for proper packing
        i4_0 = i4_0.to(tl.int32)
        i4_1 = i4_1.to(tl.int32)
        i4_2 = i4_2.to(tl.int32)
        i4_3 = i4_3.to(tl.int32)
        i4_4 = i4_4.to(tl.int32)
        i4_5 = i4_5.to(tl.int32)
        i4_6 = i4_6.to(tl.int32)
        i4_7 = i4_7.to(tl.int32)

        # Pack 8 int4 values into one int32
        packed = (
            (i4_0 & 0x0F) |
            ((i4_1 & 0x0F) << 4) |
            ((i4_2 & 0x0F) << 8) |
            ((i4_3 & 0x0F) << 12) |
            ((i4_4 & 0x0F) << 16) |
            ((i4_5 & 0x0F) << 20) |
            ((i4_6 & 0x0F) << 24) |
            ((i4_7 & 0x0F) << 28)
        )

        # Store output
        tl.store(output_ptr + offs, packed, mask=mask)


def _convert_weight_to_int4pack_for_cpu_triton(
    input: torch.Tensor, innerKTiles: int
) -> torch.Tensor:
    """
    Triton implementation of _convert_weight_to_int4pack_for_cpu.
    """
    M, N = input.shape
    # Output elements: each int32 holds 8 int4 = 4 bytes worth
    output_elements = M * N // 4
    output = torch.empty(output_elements, dtype=torch.int32, device=input.device)

    # Calculate grid
    BLOCK_SIZE = 128
    num_warps = 4
    grid = (triton.cdiv(output_elements, BLOCK_SIZE),)

    _convert_weight_to_int4pack_for_cpu_kernel[grid](
        input,
        output,
        M,
        N,
        innerKTiles,
        output_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return output


def _convert_weight_to_int4pack_for_cpu(
    input: torch.Tensor, innerKTiles: int
) -> torch.Tensor:
    """
    FlagGems implementation of _convert_weight_to_int4pack_for_cpu.

    This operator converts a weight tensor to int4 packed format
    used by NVIDIA Tensor Cores for int4 inference.
    """
    logger.debug("GEMS _convert_weight_to_int4pack_for_cpu")

    M, N = input.shape

    # Get the packed flat output from Triton kernel
    packed_flat = _convert_weight_to_int4pack_for_cpu_triton(input, innerKTiles)

    # Calculate output dimensions according to PyTorch's format:
    # dim0 = ceil(M / 8)
    # dim1 = N * innerKTiles / 32
    # dim2 = 32
    # dim3 = (M * N / 4) / (dim0 * dim1 * dim2)
    dim0 = max(1, (M + 7) // 8)
    dim1 = N * innerKTiles // 32
    dim2 = 32
    total_elements = M * N // 4
    dim3 = total_elements // (dim0 * dim1 * dim2)

    # Handle edge case where dim3 would be 0
    if dim3 == 0:
        dim3 = 1
        dim1 = total_elements // (dim0 * dim2 * dim3)

    output_shape = (dim0, dim1, dim2, dim3)

    return packed_flat.view(output_shape)