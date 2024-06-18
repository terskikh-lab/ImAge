import logging
import numpy as np
from typing import Union
import os

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

## Image Quality Control


def percent_max(image: np.ndarray) -> Union[int, float]:
    max = np.max(image)
    return (
        np.count_nonzero(image == max) / image.size * 100
    )  ### subtract one or something, since it isn't detecting saturation it seems
