import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _nested_tensor_from_tensor_list(
    list,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """Create a nested tensor from a list of tensors.

    This operator constructs a nested tensor from a list of tensors.
    Since nested tensors are a prototype feature in PyTorch, this implementation
    forwards to PyTorch's native implementation using redispatch to avoid recursion.
    """
    logger.debug("GEMS _nested_tensor_from_tensor_list")
    return torch.ops.aten._nested_tensor_from_tensor_list.default.redispatch(
        _FALLBACK_KEYSET,
        list,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )