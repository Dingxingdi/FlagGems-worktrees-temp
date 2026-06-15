import logging

import torch

from flag_gems.fused.fused_add_rms_norm import fused_add_rms_norm

logger = logging.getLogger(__name__)


def AddRMSNorm(x, residual, normalized_shape, weight, eps=1e-5):
    """Fused add and RMSNorm operation.

    This function performs fused residual addition and RMS normalization.
    Both `x` and `residual` tensors will be modified in place.

    Args:
        x: Input tensor
        residual: Residual tensor to be added
        normalized_shape: Shape for RMSNorm normalization
        weight: Weight tensor for RMSNorm
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (output tensor, updated residual tensor)
    """
    logger.debug("GEMS Add+RMSNorm FORWARD")
    return fused_add_rms_norm(x, residual, normalized_shape, weight, eps)