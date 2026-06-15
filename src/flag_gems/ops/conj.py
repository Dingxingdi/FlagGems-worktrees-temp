import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("conj"),
    key=["n_elements"],
)
@triton.jit
def conj_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load complex number as (real, imag) pair
    base = offsets * 2
    real = tl.load(in_ptr + base, mask=mask)
    imag = tl.load(in_ptr + base + 1, mask=mask)

    # Store conjugate: real stays same, imag is negated
    tl.store(out_ptr + base, real, mask=mask)
    tl.store(out_ptr + base + 1, -imag, mask=mask)


def conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS CONJ")
    if not input.is_complex():
        return input

    n_elements = input.numel()
    src = input if input.is_contiguous() else input.contiguous()
    output = torch.empty_like(src)
    in_real_ptr = torch.view_as_real(src)
    out_real_ptr = torch.view_as_real(output)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    conj_kernel[grid](in_real_ptr, out_real_ptr, n_elements)

    return output