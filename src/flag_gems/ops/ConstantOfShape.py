import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


ALL_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
ALL_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def constant_of_shape_func(out, fill_value):
    return fill_value


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def constant_of_shape_func_scalar(out, fill_value):
    return tl.full(out.shape, fill_value, out.dtype)


def constant_of_shape(shape, fill_value=0, *, dtype=None):
    """Creates a tensor filled with fill_value, where the shape is specified by the input tensor.

    Args:
        shape: A 1D tensor containing the shape of the output tensor.
        fill_value: The value to fill the output tensor with. Defaults to 0.
        dtype: The desired data type of the output tensor. If not specified,
               it will be inferred from fill_value.

    Returns:
        A tensor filled with fill_value, with shape specified by shape.
    """
    logger.debug("GEMS CONSTANT_OF_SHAPE")

    # Convert shape tensor to list if it's a tensor
    if isinstance(shape, torch.Tensor):
        shape = shape.tolist()

    # Determine dtype from fill_value if not specified
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = torch.bool
        elif isinstance(fill_value, int):
            dtype = torch.int64
        else:
            dtype = torch.get_default_dtype()

    # Create output tensor
    out = torch.empty(shape, dtype=dtype, device="cuda")

    # Convert fill_value to the output dtype to ensure proper precision
    if not isinstance(fill_value, torch.Tensor):
        fill_value = torch.tensor(fill_value, dtype=dtype, device="cuda")

    if isinstance(fill_value, torch.Tensor):
        return constant_of_shape_func(out, fill_value, out0=out)
    else:
        return constant_of_shape_func_scalar(out, fill_value, out0=out)