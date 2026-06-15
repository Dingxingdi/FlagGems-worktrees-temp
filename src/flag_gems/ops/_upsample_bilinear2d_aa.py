import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

device = device.name

logger = logging.getLogger(__name__)


@triton.autotune(
    configs=runtime.get_tuned_config("upsample_bilinear2d_aa"),
    key=["N", "C", "OH", "OW"],
)
@triton.jit
def general_interpolate_bilinear2d_aa_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    pid_x = tle.program_id(axis=0)
    pid_y = tle.program_id(axis=1)
    ow = (pid_x * BLOCK_X + tl.arange(0, BLOCK_X)) % OW
    oh = (pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)) % OH

    support_w = 2 * reciprocal_scale_w if (reciprocal_scale_w >= 1.0) else 2.0
    support_h = 2 * reciprocal_scale_h if (reciprocal_scale_h >= 1.0) else 2.0

    interpolate_w = (support_w + 0.5).to(tl.int32) * 2 + 1
    interpolate_h = (support_h + 0.5).to(tl.int32) * 2 + 1

    center_w = (ow + 0.5) * reciprocal_scale_w
    center_h = (oh + 0.5) * reciprocal_scale_h

    span_start_w = tl.maximum(center_w - support_w + 0.5, 0).to(tl.int32)
    span_start_h = tl.maximum(center_h - support_h + 0.5, 0).to(tl.int32)
    span_size_w = (tl.minimum(center_w + support_w + 0.5, IW) - span_start_w).to(
        tl.int32
    )
    span_size_h = (tl.minimum(center_h + support_h + 0.5, IH) - span_start_h).to(
        tl.int32
    )

    invscale_w = 1.0 / reciprocal_scale_w if (reciprocal_scale_w >= 1.0) else 1.0
    invscale_h = 1.0 / reciprocal_scale_h if (reciprocal_scale_h >= 1.0) else 1.0
    start_minus_center_w = span_start_w - center_w
    start_minus_center_h = span_start_h - center_h

    for n in range(0, N, 1):
        for c in range(0, C, 1):
            offset_base = (
                (n * C + c) * IH + span_start_h[:, None]
            ) * IW + span_start_w[None, :]
            weight_y_total = tl.zeros((BLOCK_Y,), dtype=tl.float32)
            result = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)

            for y in range(0, interpolate_h, 1):
                wy = tl.abs((y + start_minus_center_h + 0.5) * invscale_h)
                weight_y = tl.where(
                    y < span_size_h,
                    tl.where(wy < 1.0, 1.0 - wy, 0.0),
                    0.0,
                )
                weight_y_total += weight_y
                weight_x_total = tl.zeros((BLOCK_X,), dtype=tl.float32)
                buffer = tl.zeros((BLOCK_Y, BLOCK_X), dtype=tl.float32)

                for x in range(0, interpolate_w, 1):
                    wx = tl.abs((x + start_minus_center_w + 0.5) * invscale_w)
                    weight_x = tl.where(
                        x < span_size_w,
                        tl.where(wx < 1.0, 1.0 - wx, 0.0),
                        0.0,
                    )
                    weight_x_total += weight_x
                    data = tl.load(
                        ptr_i + (offset_base + y * IW + x),
                        mask=(span_start_h[:, None] + y < IH)
                        & (span_start_w[None, :] + x < IW),
                        other=0.0,
                    )
                    buffer += data * weight_x[None, :]

                weight_x_total = tl.where(weight_x_total != 0, weight_x_total, 1.0)
                result += buffer / weight_x_total[None, :] * weight_y[:, None]

            weight_y_total = tl.where(weight_y_total != 0, weight_y_total, 1.0)
            result /= weight_y_total[:, None]
            offset_o = ((n * C + c) * OH + oh[:, None]) * OW + ow[None, :]
            tl.store(ptr_o + offset_o, result)


def bilinear_reciprocal_scale(src_size, dst_size, align_corners, scale):
    if align_corners:
        if dst_size > 1:
            return (src_size - 1) / (dst_size - 1)
        else:
            return 0
    else:
        if scale is not None and scale > 0:
            return 1.0 / scale
        else:
            return src_size / dst_size


def _upsample_bilinear2d_aa(
    input: torch.Tensor,
    output_size: Tuple[int],
    align_corners: bool = False,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    logger.debug("GEMS UPSAMPLE BILINEAR2D AA")
    assert input.device.type == device
    assert input.ndim == 4, "The ndim of input must be 4"
    assert len(output_size) == 2, "The len of output_size must be 2"

    OH, OW = output_size
    N, C, IH, IW = input.shape

    reciprocal_scale_h = bilinear_reciprocal_scale(IH, OH, align_corners, scales_h)
    reciprocal_scale_w = bilinear_reciprocal_scale(IW, OW, align_corners, scales_w)

    # allocate output
    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)
    grid = lambda META: (
        triton.cdiv(OW, META["BLOCK_X"]),
        triton.cdiv(OH, META["BLOCK_Y"]),
    )
    # Always use the general kernel for correctness
    # The general kernel handles both upsample and downsample cases
    kernel = general_interpolate_bilinear2d_aa_kernel
    with torch_device_fn.device(input.device):
        kernel[grid](
            output,
            input,
            N,
            C,
            OH,
            OW,
            IH,
            IW,
            reciprocal_scale_h,
            reciprocal_scale_w,
        )
    return output