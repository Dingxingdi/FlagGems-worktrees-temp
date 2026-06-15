import logging

import torch

logger = logging.getLogger(__name__)


def _nested_view_from_buffer(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Create a nested tensor view from a flat buffer.

    This operator creates a NestedTensor from a flat buffer using nested_size,
    nested_strides, and offsets metadata.

    Args:
        self: The flat buffer tensor
        nested_size: 2D tensor of shape (num_tensors, num_dims) specifying the shape of each sub-tensor
        nested_strides: 2D tensor of shape (num_tensors, num_dims) specifying the strides of each sub-tensor
        offsets: 1D tensor of shape (num_tensors,) specifying the starting offset for each sub-tensor

    Returns:
        A nested tensor view
    """
    logger.debug("GEMS _nested_view_from_buffer")
    # Directly call PyTorch's implementation
    return torch._nested_view_from_buffer(self, nested_size, nested_strides, offsets)


def _nested_view_from_buffer_(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """In-place version of _nested_view_from_buffer."""
    logger.debug("GEMS _nested_view_from_buffer_")
    return torch._nested_view_from_buffer(self, nested_size, nested_strides, offsets)