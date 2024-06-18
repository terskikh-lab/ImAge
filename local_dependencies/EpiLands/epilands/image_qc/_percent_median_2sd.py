import logging
import numpy as np
from typing import Union
import os

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

## Image Quality Control


def percent_median2SD(image: np.ndarray) -> Union[int, float]:
    med = np.median(image)
    std = np.std(image)
    return np.count_nonzero(image > (med + 2 * std)) / image.size * 100
