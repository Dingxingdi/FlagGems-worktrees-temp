import logging

from flag_gems.ops.atan import atan as _atan
from flag_gems.ops.atan import atan_ as _atan_

logger = logging.getLogger(__name__)


def arctan(A):
    logger.debug("GEMS ARCTAN")
    return _atan(A)


def arctan_(A):
    logger.debug("GEMS ARCTAN_")
    return _atan_(A)