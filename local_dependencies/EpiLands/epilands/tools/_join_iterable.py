import logging
import numpy as np

# RELATIVE IMPORTS #
from ._check_if_str import check_if_str
from ..config import COLUMN_SEPARATOR


def join_iterable(t):
    if isinstance(t, (list, tuple, np.ndarray)):
        if len(t) == 1:
            return str(t[0])
        elif len(t) > 1:
            t = [check_if_str(i) for i in t if i != None]
            return COLUMN_SEPARATOR.join(t)
        else:
            logging.error("{} was empty: {}".format(type(t), t))
    if isinstance(t, (str, type(None))):
        return t
    if isinstance(t, (float, int)):
        return str(t)
    else:
        raise ValueError(
            f"join_tuple only accepts types {[list, tuple, str]} but {type(t)} was given"
        )
