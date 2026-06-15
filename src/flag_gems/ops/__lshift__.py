import logging

import flag_gems
from flag_gems.ops.bitwise_left_shift import bitwise_left_shift

logger = logging.getLogger(__name__)


def __lshift__(self, other):
    logger.debug("GEMS __lshift__")
    return bitwise_left_shift(self, other)