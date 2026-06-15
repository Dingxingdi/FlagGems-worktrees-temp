import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.heuristics(runtime.get_heuristic_config("dropout"))
@triton.jit(do_not_specialize=["scale"])
def native_dropout_backward_kernel(
    DY,
    DX,
    dropout_mask,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    m = tl.load(
        dropout_mask + offset, mask=mask, other=0, eviction_policy="evict_first"
    )
    dy = tl.load(DY + offset, mask=mask, other=0, eviction_policy="evict_first")
    dx = dy * m * scale
    tl.store(DX + offset, dx, mask=mask, eviction_policy="evict_first")


def native_dropout_backward(grad_output, mask, scale):
    logger.debug("GEMS NATIVE_DROPOUT_BACKWARD")
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(grad_output)
    N = grad_output.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    with torch_device_fn.device(grad_output.device):
        native_dropout_backward_kernel[grid_fn](
            grad_output, grad_input, mask, N, scale
        )
    return grad_input