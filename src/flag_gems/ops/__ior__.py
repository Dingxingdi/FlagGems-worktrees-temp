import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def __ior___func(x, y):
    return x | y


def __ior__(A, B):
    logger.debug("GEMS __ior__")
    return __ior___func(A, B)


def __ior__tensor_(A, B):
    logger.debug("GEMS __ior___")
    return __ior___func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def __ior___func_scalar(x, y):
    return x | y


def __ior__scalar(A, B):
    logger.debug("GEMS __ior__ SCALAR")
    return __ior___func_scalar(A, B)


def __ior__scalar_(A, B):
    logger.debug("GEMS __ior___ SCALAR")
    return __ior___func_scalar(A, B, out0=A)


def __ior__scalar_tensor(A, B):
    logger.debug("GEMS __ior__ SCALAR TENSOR")
    return __ior___func_scalar(B, A)