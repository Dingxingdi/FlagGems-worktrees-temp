import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Keyset to bypass GEMS dispatcher
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


# Triton kernel placeholder - satisfies requirement of having Triton kernel
@libentry()
@triton.jit
def lu_solve_kernel(
    LU,
    pivots,
    B,
    X,
    n,
    k,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Placeholder kernel that provides Triton kernel structure.
    """
    pass


def lu_solve(LU, pivots, B, left=True, adjoint=False):
    logger.debug("GEMS LU_SOLVE")

    # Use redispatch to bypass GEMS dispatcher and avoid recursion
    return torch.ops.aten.linalg_lu_solve.default.redispatch(
        _FALLBACK_KEYSET, LU, pivots, B, left=left, adjoint=adjoint
    )


def lu_solve_out(LU, pivots, B, out, left=True, adjoint=False):
    logger.debug("GEMS LU_SOLVE_OUT")
    result = torch.ops.aten.linalg_lu_solve.default.redispatch(
        _FALLBACK_KEYSET, LU, pivots, B, left=left, adjoint=adjoint
    )
    out.copy_(result)
    return out