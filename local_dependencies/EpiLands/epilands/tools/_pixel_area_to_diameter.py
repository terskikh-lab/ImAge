import numpy as np
from typing import Union


def pixel_area_to_diameter(x: Union[float, int]) -> float:
    return np.sqrt(4 * x / np.pi)
