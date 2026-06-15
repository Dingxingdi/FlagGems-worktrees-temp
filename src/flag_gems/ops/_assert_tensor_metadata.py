import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def _assert_tensor_metadata(
    inp: torch.Tensor,
    size: Optional[List[int]] = None,
    stride: Optional[List[int]] = None,
    dtype: Optional[torch.dtype] = None,
    *,
    device: Optional[torch.device] = None,
    layout: Optional[torch.layout] = None,
):
    """
    Assert that a tensor has the specified metadata.

    This operator checks that the input tensor matches the expected metadata
    (size, stride, dtype, device, layout) and raises an error if it doesn't.
    """
    logger.debug("GEMS _ASSERT_TENSOR_METADATA")

    # Validate size if provided
    if size is not None:
        expected_size = torch.Size(size)
        if inp.size() != expected_size:
            raise RuntimeError(
                f"Expected tensor size {expected_size}, but got {inp.size()}"
            )

    # Validate stride if provided
    if stride is not None:
        expected_stride = tuple(stride)
        if inp.stride() != expected_stride:
            raise RuntimeError(
                f"Expected tensor stride {expected_stride}, but got {inp.stride()}"
            )

    # Validate dtype if provided
    if dtype is not None:
        if inp.dtype != dtype:
            raise RuntimeError(
                f"Expected tensor dtype {dtype}, but got {inp.dtype}"
            )

    # Validate device if provided
    if device is not None:
        if inp.device != device:
            raise RuntimeError(
                f"Expected tensor device {device}, but got {inp.device}"
            )

    # Validate layout if provided
    if layout is not None:
        if inp.layout != layout:
            raise RuntimeError(
                f"Expected tensor layout {layout}, but got {inp.layout}"
            )

    return None