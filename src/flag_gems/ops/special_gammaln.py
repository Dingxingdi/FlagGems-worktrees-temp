import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_lgamma = tl_extra_shim.lgamma

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def special_gammaln_func(x):
    # gammaln(x) = ln(|Γ(x)|)
    # lgamma in libdevice already computes ln(|gamma(x)|)
    xf = x.to(tl.float32)
    return _lgamma(xf).to(x.dtype)


def special_gammaln(A):
    logger.debug("GEMS SPECIAL_GAMMALN")
    return special_gammaln_func(A)


def special_gammaln_out(A, *, out=None):
    logger.debug("GEMS SPECIAL_GAMMALN_OUT")
    if out is None:
        return special_gammaln_func(A)
    special_gammaln_func(A, out0=out)
    return out