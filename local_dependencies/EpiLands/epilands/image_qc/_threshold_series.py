import logging
import numpy as np
import pandas as pd
from typing import Union
import os

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

## Image Quality Control


def threshold_series(
    size_series: pd.Series,
    size_thresh_low: Union[int, float],
    size_thresh_high: Union[int, float],
) -> np.ndarray:
    """excludes objects of a particular value given high and low thresholds"""
    logger.info(
        f"Thresholding objects by size: {size_thresh_low} < size < {size_thresh_high}"
    )
    bool_mask = (size_series.values > size_thresh_low) & (
        size_series.values < size_thresh_high
    )
    in_objects = size_series[bool_mask].index.values
    out_objects = size_series[~bool_mask].index.values
    return in_objects, out_objects
