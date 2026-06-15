import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def log_sigmoid_backward_func(grad_output, self, buffer):
    # buffer = exp(-self) from log_sigmoid_forward
    # gradient = grad_output * buffer / (1 + buffer)
    #            = grad_output * exp(-self) / (1 + exp(-self))
    #            = grad_output * (1 - sigmoid(self))
    # Compute in float32 for better precision, then convert back
    grad_output_fp32 = grad_output.to(tl.float32)
    buffer_fp32 = buffer.to(tl.float32)
    result = grad_output_fp32 * buffer_fp32 / (1.0 + buffer_fp32)
    return result.to(grad_output.dtype)


def log_sigmoid_backward(grad_output, self, buffer):
    logger.debug("GEMS LOG_SIGMOID_BACKWARD")
    return log_sigmoid_backward_func(grad_output, self, buffer)


def log_sigmoid_backward_(grad_output, self, buffer):
    logger.debug("GEMS LOG_SIGMOID_BACKWARD_")
    return log_sigmoid_backward_func(grad_output, self, buffer, out0=grad_output)