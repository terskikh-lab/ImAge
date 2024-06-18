from typing import Iterable
import numpy as np


def reorder_iterable(iterable: Iterable, order: Iterable) -> np.ndarray:
    """
    reorders an iterable according to the order given

    iterable: list the iterable to reorder

    order: list the order to use for reordering the iterable
    """
    return np.array([iterable[i - 1] for i in order])
