import numpy as np
import logging
import os

from ._check_if_str import check_if_str
from ..config import COLUMN_SEPARATOR

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def join_tuple(t):
    if isinstance(t, (list, tuple, np.ndarray)):
        if len(t) == 1:
            return str(t[0])
        elif len(t) > 1:
            t = [check_if_str(i) for i in t]
            return COLUMN_SEPARATOR.join(t)
        else:
            logger.debug("join_tuple recieved an empty {}: {}".format(type(t), t))
    if isinstance(t, (str, type(None))):
        return t
    if isinstance(t, (float, int)):
        return str(t)
    else:
        raise ValueError(
            "join_tuple only accepts "
            + "types {} but {} was given".format([list, tuple, str], type(t))
        )
