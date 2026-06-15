import logging

import torch
from torch import Tensor

from flag_gems.ops.batch_norm import batch_norm

logger = logging.getLogger(__name__)


def _native_batch_norm_legit_no_training(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
) -> tuple[Tensor, Tensor, Tensor]:
    logger.debug("GEMS _native_batch_norm_legit_no_training")

    # Call batch_norm with training=False (inference mode)
    # batch_norm returns (output, mean, inv_std) where mean and inv_std are
    # loaded from running_mean/running_var in inference mode
    output, mean, inv_std = batch_norm(
        input=input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=False,
        momentum=momentum,
        eps=eps,
    )

    # _native_batch_norm_legit_no_training returns empty tensors for mean and var
    # to match PyTorch behavior
    empty_mean = torch.empty(0, dtype=input.dtype, device=input.device)
    empty_var = torch.empty(0, dtype=input.dtype, device=input.device)

    return output, empty_mean, empty_var