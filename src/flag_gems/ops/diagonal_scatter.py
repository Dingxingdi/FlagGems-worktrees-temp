import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    logger.debug("GEMS DIAGONAL_SCATTER")
    # Create output by cloning input (fresh storage, not a view)
    output = input.clone()
    # Get the diagonal view of the output tensor
    diag = torch.diagonal(output, offset, dim1, dim2)
    # Copy src values into the diagonal view
    # The src tensor has the same shape as the diagonal view
    copy_func.instantiate(src.ndim)(src, out0=diag)
    return output


def diagonal_scatter_(input, src, offset=0, dim1=0, dim2=1):
    logger.debug("GEMS DIAGONAL_SCATTER_")
    # In-place version: modify input directly
    # Get the diagonal view of the input tensor
    diag = torch.diagonal(input, offset, dim1, dim2)
    # Copy src values into the diagonal view
    copy_func.instantiate(src.ndim)(src, out0=diag)
    return input