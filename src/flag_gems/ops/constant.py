import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def constant_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


def constant(input, value):
    """Fill the input tensor with a constant value.

    Args:
        input: The input tensor to fill
        value: The constant value to fill with

    Returns:
        A tensor filled with the constant value
    """
    logger.debug("GEMS CONSTANT")
    out = torch.empty_like(input)
    return constant_func(input, value, out0=out)


def constant_(self, value):
    """In-place version: fill the tensor with a constant value.

    Args:
        self: The tensor to fill in-place
        value: The constant value to fill with

    Returns:
        The tensor filled with the constant value
    """
    logger.debug("GEMS CONSTANT_")
    constant_func(self, value, out0=self)
    return self