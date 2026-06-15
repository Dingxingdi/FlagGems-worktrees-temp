import logging

import torch

logger = logging.getLogger(__name__)

# Get the original PyTorch implementation before flag_gems registration
_original_grid_sampler_3d_backward = torch.ops.aten.grid_sampler_3d_backward.default


def grid_sampler_3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
    output_mask: tuple = (True, True),
):
    """
    Backward pass for grid_sampler_3d.
    Uses original PyTorch implementation to avoid recursion.
    """
    logger.debug("GEMS grid_sampler_3d_backward")

    # Call the original PyTorch implementation
    return _original_grid_sampler_3d_backward(
        grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask
    )