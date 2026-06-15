import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def channel_shuffle_kernel(
    inp,
    out,
    N,
    C,
    H,
    W,
    groups,
    channels_per_group: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    total_elements = N * H * W
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements

    n = offset // (H * W)
    hw = offset % (H * W)
    h = hw // W
    w = hw % W

    # Inverse permutation: output[c] = input[src_c]
    # src_c = (c // groups) * channels_per_group + (c % groups)
    # No wait, that's wrong. Let me use a simpler approach:
    # For output channel c, we compute c_div = c % channels_per_group, c_mod = c // channels_per_group
    # Then src_c = c_mod * groups + c_div
    for c in range(C):
        c_div = c // groups
        c_mod = c % groups
        src_c = c_mod * channels_per_group + c_div
        src_offset = ((n * C + src_c) * H + h) * W + w
        dst_offset = ((n * C + c) * H + h) * W + w
        val = tl.load(inp + src_offset)
        tl.store(out + dst_offset, val)


def channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    logger.debug("GEMS CHANNEL_SHUFFLE")
    assert input.dim() >= 3, "Input must have at least 3 dimensions (C, H, W)"
    C = input.shape[-3]
    H = input.shape[-2]
    W = input.shape[-1]
    N = input.numel() // (C * H * W)

    assert C % groups == 0, "Number of channels must be divisible by groups"
    channels_per_group = C // groups

    out = torch.empty_like(input)
    input = input.contiguous()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N * H * W, BLOCK_SIZE),)

    with torch_device_fn.device(input.device):
        channel_shuffle_kernel[grid](
            input,
            out,
            N,
            C,
            H,
            W,
            groups,
            channels_per_group,
            BLOCK_SIZE,
        )
    return out