import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def as_strided_copy(self, size, stride, storage_offset=None):
    logger.debug("GEMS AS_STRIDED_COPY")
    if storage_offset is None:
        storage_offset = 0

    # Create a StridedBuffer that reinterprets the input tensor with new size/stride
    src_buffer = StridedBuffer(
        base=self,
        shape=size,
        strides=stride,
        dtype=self.dtype,
        offset=storage_offset,
    )

    # Allocate output tensor with the target shape
    out = torch.empty(size, dtype=self.dtype, device=self.device)

    # Use the copy kernel to copy from the strided buffer to the output
    overload = copy_func.instantiate(len(size))
    overload(src_buffer, out0=out)
    return out