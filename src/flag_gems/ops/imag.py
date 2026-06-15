import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("imag"),
    key=["n_elements"],
)
@triton.jit
def imag_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Complex tensors are stored as [real0, imag0, real1, imag1, ...]
    # We need to load the imaginary parts (every other element starting at index 1)
    base = offsets * 2
    imag_vals = tl.load(in_ptr + base + 1, mask=mask)

    tl.store(out_ptr + offsets, imag_vals, mask=mask)


def imag(A):
    logger.debug("GEMS IMAG")
    if not A.is_complex():
        raise ValueError("imag is only supported for complex tensors")

    n_elements = A.numel()
    src = A if A.is_contiguous() else A.contiguous()

    # Get the underlying real representation without copying
    # This gives us a view of the data as (n, 2) real numbers
    in_real = torch.view_as_real(src)

    # The output should have the same dtype as the real part
    output = torch.empty_like(src.real)

    # Now we need to extract just the imaginary part (column index 1)
    # We can do this by loading from in_real with stride 2 and offset 1
    # But Triton kernel expects a 1D pointer, so let's use a modified approach

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    imag_kernel[grid](in_real, output, n_elements)

    # Reshape to match input shape
    return output.reshape(A.shape)