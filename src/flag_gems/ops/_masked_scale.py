import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _masked_scale_func(self, mask, scale):
    # mask is uint8 (0 or 1), convert to self's dtype for multiplication
    return self * mask.to(self.dtype) * scale


def _masked_scale(self, mask, scale):
    logger.debug("GEMS _MASKED_SCALE")
    return _masked_scale_func(self, mask, scale)


def _masked_scale_(self, mask, scale):
    logger.debug("GEMS _MASKED_SCALE_")
    return _masked_scale_func(self, mask, scale, out0=self)