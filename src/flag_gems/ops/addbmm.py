import logging

import torch

logger = logging.getLogger(__name__)


def addbmm(bias, batch1, batch2, beta=1.0, alpha=1.0):
    logger.debug("GEMS ADDBMM")

    # Compute in float32 for numerical stability, then convert back
    if bias.dtype in (torch.float16, torch.bfloat16):
        # Cast to float32 for computation
        bias_fp32 = bias.to(torch.float32)
        batch1_fp32 = batch1.to(torch.float32)
        batch2_fp32 = batch2.to(torch.float32)

        # Compute batch matrix multiplication: (batch, M, K) @ (batch, K, N) -> (batch, M, N)
        # Use torch.ops.aten.bmm to avoid recursion through flag_gems
        bmm_result = torch.ops.aten.bmm(batch1_fp32, batch2_fp32)
        # Sum over batch dimension: (batch, M, N) -> (M, N)
        sum_result = bmm_result.sum(dim=0)
        # Compute final result: beta * bias + alpha * sum_result
        result = beta * bias_fp32 + alpha * sum_result
        # Convert back to original dtype
        result = result.to(bias.dtype)
    else:
        # For float32 and float64, use direct computation
        bmm_result = torch.ops.aten.bmm(batch1, batch2)
        sum_result = bmm_result.sum(dim=0)
        result = beta * bias + alpha * sum_result

    return result


def addbmm_(bias, batch1, batch2, beta=1.0, alpha=1.0):
    logger.debug("GEMS ADDBMM_")
    result = addbmm(bias, batch1, batch2, beta=beta, alpha=alpha)
    bias.copy_(result)
    return bias