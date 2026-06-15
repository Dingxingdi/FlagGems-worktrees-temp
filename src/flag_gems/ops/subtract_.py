import logging

from flag_gems.ops.sub import sub_

logger = logging.getLogger(__name__)


def subtract_(A, B, *, alpha=1):
    """In-place subtraction: A.subtract_(B, alpha=alpha) is equivalent to A -= B * alpha"""
    logger.debug("GEMS SUBTRACT_")
    return sub_(A, B, alpha=alpha)