"""
Implementation of igammac (regularized incomplete gamma function complement).
igammac(a, x) = Q(a, x) = 1 - P(a, x) = gammaincc(a, x)
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

# Get math functions from tl_extra_shim for cross-backend compatibility
_exp = tl_extra_shim.exp
_pow = tl_extra_shim.pow


@triton.jit
def igamma_lower_series(a, x):
    """
    Compute lower regularized incomplete gamma function P(a, x) using series expansion.
    P(a, x) = e^(-x) * x^a * Σ(n=0 to ∞) [x^n / (a(a+1)...(a+n))]
    Converges rapidly for x < a + 1.
    """
    # Initialize
    result = 1.0 / a
    term = result

    # Series expansion - more iterations for better accuracy
    for n in range(1, 50):
        term = term * x / (a + n)
        result = result + term

    # Multiply by e^(-x) * x^a
    result = result * _exp(-x) * _pow(x, a)

    return result


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def igammac_func(a, x):
    """
    Compute regularized incomplete gamma function complement Q(a, x).
    Q(a, x) = 1 - P(a, x) = igammac(a, x)

    Uses series expansion to compute P, then Q = 1 - P.
    Works best when x < a + 1.
    """
    # Convert to float32 for computation
    a_f32 = a.to(tl.float32)
    x_f32 = x.to(tl.float32)

    # Compute P(a, x) using series expansion
    p = igamma_lower_series(a_f32, x_f32)

    # Q(a, x) = 1 - P(a, x)
    q = 1.0 - p

    # Clamp to [0, 1] for numerical stability
    q = tl.where(q < 0.0, 0.0, q)
    q = tl.where(q > 1.0, 1.0, q)

    # Convert back to original dtype
    return q.to(a.dtype)


def igammac(a, b):
    """Compute the regularized incomplete gamma function complement."""
    logger.debug("GEMS IGAMMAC")
    return igammac_func(a, b)


def igammac_(a, b):
    """In-place version of igammac."""
    logger.debug("GEMS IGAMMAC_")
    return igammac_func(a, b, out0=a)