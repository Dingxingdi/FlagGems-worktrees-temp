import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_kernel(a, b):
    return a << b


def bitwise_left_shift(self, other, *, out=None):
    logger.debug("GEMS BITWISE_LEFT_SHIFT")
    return bitwise_left_shift_kernel(self, other, out=out)


def bitwise_left_shift_(self, other):
    logger.debug("GEMS BITWISE_LEFT_SHIFT_")
    return bitwise_left_shift_kernel(self, other, out0=self)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_kernel_scalar(a, b):
    return a << b


def bitwise_left_shift_scalar(self, other, *, out=None):
    logger.debug("GEMS BITWISE_LEFT_SHIFT_SCALAR")
    return bitwise_left_shift_kernel_scalar(self, other, out=out)


def bitwise_left_shift_scalar_(self, other):
    logger.debug("GEMS BITWISE_LEFT_SHIFT_SCALAR_")
    return bitwise_left_shift_kernel_scalar(self, other, out0=self)
