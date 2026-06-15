import logging
from typing import Optional

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _resize_as_kernel(src):
    return src


def _can_use_triton(tensor: torch.Tensor) -> bool:
    if tensor.layout != torch.strided:
        return False
    if tensor.is_quantized:
        return False
    if tensor.is_complex():
        return False
    return True


def resize_as(self: torch.Tensor, the_template: torch.Tensor, *,
              memory_format: Optional[torch.memory_format] = None):
    logger.debug("GEMS RESIZE_AS")

    if memory_format is None:
        memory_format = torch.contiguous_format

    # Check if already same shape
    if self.shape == the_template.shape:
        return self

    # Validate element count (PyTorch requires same number of elements for autograd)
    if self.numel() != the_template.numel():
        raise RuntimeError(
            f"requested resize to {tuple(the_template.shape)} ({the_template.numel()} elements in total), "
            f"but the given tensor has a size of {tuple(self.shape)} ({self.numel()} elements in total). "
            f"autograd's resize can only change the shape of a given tensor, while preserving the number of elements."
        )

    if not _can_use_triton(the_template):
        return torch.ops.aten.resize_as.default.redispatch(
            _FALLBACK_KEYSET, self, the_template, memory_format=memory_format
        )

    # Create output tensor with target shape
    out = torch.empty_strided(
        the_template.size(),
        the_template.stride(),
        dtype=the_template.dtype,
        device=the_template.device
    )

    # Expand source to match destination shape for broadcast
    if self.shape != the_template.shape:
        expanded = self.expand(the_template.shape)
    else:
        expanded = self

    # Handle empty tensors
    if expanded.numel() == 0:
        return out

    overload = _resize_as_kernel.instantiate(expanded.ndim)
    overload(expanded, out0=out)
    return out


def resize_as_(self: torch.Tensor, the_template: torch.Tensor, *,
               memory_format: Optional[torch.memory_format] = None):
    logger.debug("GEMS RESIZE_AS_")

    if memory_format is None:
        memory_format = torch.contiguous_format

    # Check if already same shape
    if self.shape == the_template.shape:
        return self

    # Validate element count
    if self.numel() != the_template.numel():
        raise RuntimeError(
            f"requested resize to {tuple(the_template.shape)} ({the_template.numel()} elements in total), "
            f"but the given tensor has a size of {tuple(self.shape)} ({self.numel()} elements in total). "
            f"autograd's resize can only change the shape of a given tensor, while preserving the number of elements."
        )

    if not _can_use_triton(the_template):
        return torch.ops.aten.resize_as_.default.redispatch(
            _FALLBACK_KEYSET, self, the_template, memory_format=memory_format
        )

    # Resize in-place: change self to match template shape
    self.resize_(the_template.size())

    # If shapes differ, we need to copy data (but since it's in-place resize,
    # the data is already there, just reinterpreted)
    return self