import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def complex_kernel(real, imag):
    return real, imag


def complex(real, imag):
    logger.debug("GEMS COMPLEX")
    if real.shape != imag.shape:
        raise ValueError(
            f"real and imag must have the same shape, got {real.shape} and {imag.shape}"
        )
    if real.dtype != imag.dtype:
        raise ValueError(
            f"real and imag must have the same dtype, got {real.dtype} and {imag.dtype}"
        )

    # Handle bfloat16: convert to float32 for view_as_complex compatibility
    input_dtype = real.dtype
    if input_dtype == torch.bfloat16:
        real = real.to(torch.float32)
        imag = imag.to(torch.float32)

    # Create output tensor with extra dimension for real/imag parts
    output = torch.empty((*real.shape, 2), dtype=real.dtype, device=real.device)
    complex_kernel(real, imag, out0=output[..., 0], out1=output[..., 1])
    result = torch.view_as_complex(output)

    # Convert back to original dtype if needed
    if input_dtype == torch.bfloat16:
        result = result.to(torch.complex32)

    return result