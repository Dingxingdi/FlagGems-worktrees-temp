import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def swiglu_forward_kernel(a, b):
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    a_fp32 = a.to(tl.float32)
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a_fp32 * b.to(tl.float32) * sigmoid_b
    return result


@pointwise_dynamic(
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ]
)
@triton.jit
def swiglu_backward_kernel(grad_output, a, b):
    # Forward: output = a * silu(b) = a * b * sigmoid(b)
    # silu(b) = b * sigmoid(b)
    # d(silu(b))/db = sigmoid(b) + b * sigmoid(b) * (1 - sigmoid(b))
    #               = sigmoid(b) * (1 + b * (1 - sigmoid(b)))
    b_fp32 = b.to(tl.float32)
    a_fp32 = a.to(tl.float32)
    go_fp32 = grad_output.to(tl.float32)
    sigmoid_b = 1 / (1 + exp(-b_fp32))
    da = go_fp32 * sigmoid_b * b_fp32
    db = go_fp32 * a_fp32 * sigmoid_b * (1 + b_fp32 * (1 - sigmoid_b))

    return da, db


def Fused_SwiGLU(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS Fused_SwiGLU FORWARD")
    # Split into a and b along the specified dimension
    a, b = torch.chunk(self, 2, dim=dim)
    out = swiglu_forward_kernel(a, b)

    return out


def Fused_SwiGLU_backward(grad_output, self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GEMS Fused_SwiGLU BACKWARD")
    # Recreate a and b
    a, b = torch.chunk(self, 2, dim=dim)
    grad_input = torch.empty_like(self, memory_format=torch.contiguous_format)
    grad_a, grad_b = torch.chunk(grad_input, 2, dim=dim)
    swiglu_backward_kernel(grad_output, a, b, out0=grad_a, out1=grad_b)

    return grad_input