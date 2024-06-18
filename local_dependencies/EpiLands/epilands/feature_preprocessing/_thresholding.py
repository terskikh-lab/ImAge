# Import libraries
from __future__ import annotations
import pandas as pd
import numpy as np

# relative imports
from ..math.conversions import pixel_area_to_diameter


def size_threshold_df(
    df: pd.DataFrame,
    high_threshold: float = 2.0,
    low_threshold: float = 0,
    threshold_col: str = "Ch1_MOR_Nucleus_area",
    threshold_metric: str = "median",
    convert_area_to_diameter: bool = False,
    **kwargs,
) -> pd.Series:
    """
    This function takes a dataframe of single cell data and eliminates doublets based on the median of the threshold_col.

    df: pd.DataFrame - the dataframe of single cell data
    high_threshold: float - the threshold for the high thresholding
    low_threshold: float - the threshold for the low thresholding
    threshold_col: str - the column in the dataframe to use for thresholding
    threshold_metric: str - the metric to use for thresholding
    units: str - the units of the threshold_col
    show_histogram: bool - whether or not to show a histogram of the threshold_col
    display_quantile: float - the quantile to use for displaying the threshold_col
    convert_area_to_diameter: bool - whether or not to convert the threshold_col from pixel area to diameter
    """
    if not isinstance(high_threshold, (int, float)):
        raise ValueError(
            "high_threshold must be a number not {}".format(type(high_threshold))
        )
    if not isinstance(low_threshold, (int, float)):
        raise ValueError(
            "low_threshold must be a number not {}".format(type(low_threshold))
        )
    if high_threshold == None or low_threshold == None:
        raise ValueError(
            "threshold_objects: high_threshold or low_threshold was left undefined."
            + " If none, please use high threshold = np.inf or low_threshold = 0"
        )
    if threshold_metric not in ["median", "mean", "values", "quantile"]:
        raise ValueError(
            "threshold_objects: threshold metric must be one of"
            + " {} but {} was given".format(
                ["median", "mean", "values", "quantile"], threshold_metric
            )
        )

    data_series = df[threshold_col].astype(float)  # copy the input data and reset index
    if convert_area_to_diameter == True:
        data_series = data_series.map(pixel_area_to_diameter)
    if threshold_metric == "median":
        high_threshold = high_threshold * data_series.median()
        low_threshold = low_threshold * data_series.median()
    elif threshold_metric == "mean":
        high_threshold = high_threshold * data_series.mean()
        low_threshold = low_threshold * data_series.mean()
    elif threshold_metric == "quantile":
        high_threshold = data_series.quantile(high_threshold)
        low_threshold = data_series.quantile(low_threshold)
    object_count = len(data_series)  # count the number of rows in the dataframe
    # display_range = (int(0), int(data_series.quantile(display_quantile)))
    print(
        "Starting with", object_count, "objects"
    )  # print the number of rows in the dataframe
    print("Before Thresholding:")
    print(
        data_series.describe(percentiles=[0.5, 0.75, 0.9, 0.95])
    )  # print the summary statistics of the threshold_col
    # if show_histogram == True: #if the user wants to see the size distribution graphs
    #     threshold_objects_histogram(
    #         data_series,
    #         high_threshold,
    #         low_threshold,
    #         display_range,
    #         cells_per_bin=200,
    #         xlabel = ' '.join((threshold_col, units)),
    #         **kwargs
    #         )
    inlier_objects = (data_series < high_threshold) & (
        data_series > low_threshold
    )  # create a series of cells that are inside the boundaries
    print("After Thresholding:")
    print(
        data_series[inlier_objects].describe(percentiles=[0.5, 0.75, 0.9, 0.95])
    )  # print the summary statistics of the threshold_col
    # if show_histogram == True: #if the user wants to see the size distribution graphs
    #     threshold_objects_histogram(
    #         data_series[inlier_objects],
    #         high_threshold,
    #         low_threshold,
    #         display_range,
    #         cells_per_bin=200,
    #         xlabel = ' '.join((threshold_col, units)),
    #         **kwargs
    #         )
    outlier_count = inlier_objects.to_list().count(False)
    # print the number of cells dropped and the percentage of cells dropped
    print(
        "Found {} objects ({}%) outside of the size boundaries ({},{})".format(
            outlier_count,
            np.round(outlier_count / object_count * 100, 3),
            low_threshold,
            high_threshold,
        )
    )
    return inlier_objects  # return a boolean mask for outlier cells
