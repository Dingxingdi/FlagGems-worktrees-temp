import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.randn import randn_kernel
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)
UNROLL = 4


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def exp_func(x):
    return tl.exp(x.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func(val, std, mean):
    return val * std + mean


def log_normal_distribution(shape, device, dtype, *, generator=None, out=None):
    if generator is not None:
        raise NotImplementedError("Custom generator is not supported")
    if out is None:
        out = torch.empty(shape, device=device, dtype=dtype)
    N = volume(shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment, generator=generator)
    with torch_device_fn.device(device):
        randn_kernel[grid_fn](out, N, philox_seed, philox_offset)
    return out


def log_normal(self, mean=1.0, std=2.0, *, generator=None):
    logger.debug("GEMS LOG_NORMAL")
    if self.numel() == 0:
        return torch.empty_like(self)
    out = log_normal_distribution(self.shape, self.device, self.dtype, generator=generator)
    # Transform: val * std + mean (normal distribution params), then exp for log-normal
    # Since normal distribution uses mean=0, std=1, we transform: val * std + mean
    out = out * std + mean
    return exp_func(out)


def log_normal_(self, mean=1.0, std=2.0, *, generator=None):
    logger.debug("GEMS LOG_NORMAL_")
    if self.numel() == 0:
        return self
    self = log_normal_distribution(self.shape, self.device, self.dtype, generator=generator, out=self)
    # Transform in-place: val * std + mean, then exp for log-normal
    transform_func(self, std, mean, out0=self)
    exp_func(self, out0=self)
    return self