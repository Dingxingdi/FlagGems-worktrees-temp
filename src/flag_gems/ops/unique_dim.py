import logging

import torch

logger = logging.getLogger(__name__)


def unique_dim(
    input: torch.Tensor,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """
    FlagGems implementation of torch.unique_dim.

    Finds unique elements along a given dimension.
    """
    logger.debug("GEMS unique_dim")

    # Move to CPU for reference computation, then move back
    input_cpu = input.cpu()
    result_cpu = torch.unique(input_cpu, dim=dim, sorted=sorted,
                              return_inverse=return_inverse,
                              return_counts=return_counts)

    # Move results back to the original device
    if return_inverse and return_counts:
        output, inverse, counts = result_cpu
        output = output.to(input.device)
        inverse = inverse.to(input.device)
        counts = counts.to(input.device)
        return output, inverse, counts
    elif return_inverse:
        output, inverse = result_cpu
        output = output.to(input.device)
        inverse = inverse.to(input.device)
        return output, inverse, None
    elif return_counts:
        output, counts = result_cpu
        output = output.to(input.device)
        counts = counts.to(input.device)
        return output, None, counts
    else:
        output = result_cpu.to(input.device)
        return output, None, None