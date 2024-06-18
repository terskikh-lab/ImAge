from __future__ import annotations
import pandas as pd
import numpy as np

from ..tools import aggregate_df, reshape_dataframe_to_plate
from ..generic_read_write import save_dataframe_to_csv


def generate_intensity_map(
    df: pd.DataFrame,
    value_col: str,
    save: bool,
    output_directory: str,
    metric: str = "mean",
    quantile: float = None,
    wellindex_col: str = "WellIndex",
) -> pd.DataFrame:
    """
    generates an intensity map from a dataframe given the wellindex column.

    df: pd.DataFrame

    value_col: str the name of the column containing the intensity values

    save: bool tells the function whether to save the cellcount map to a file

    output_directory: str where to save the cellcount map

    metric: str one of ['mean', 'median', 'quantile']
    the metric to use for downsampling from cells -> wells in the intensity map

    quantile: float the quantile to use for downsampling from cells -> wells in the intensity map if metric is 'quantile'

    wellindex_col: str the name of the column containing the wellindex
    """

    if metric not in ["mean", "median", "quantile"]:
        raise ValueError(
            "generate_intensity_map: metric must be one of {}".format(
                ["mean", "median", "quantile"]
            )
            + " But {} was given".format(metric)
        )
    if metric == "mean":
        df_intensity = df.groupby([wellindex_col])[value_col].mean().reset_index()
        name = "{}-{}".format(metric, value_col)
    if metric == "median":
        df_intensity = df.groupby([wellindex_col])[value_col].median().reset_index()
        name = "{}-{}".format(metric, value_col)
    if metric == "quantile":
        if not isinstance(quantile, float):
            raise ValueError(
                "if metric == 'quantile', the variable quantile"
                " must be set to a float, but {} was given".format(quantile)
            )
        df_intensity = (
            df.groupby([wellindex_col])[value_col].quantile(quantile).reset_index()
        )
        name = "{}-{}".format(metric, quantile)
    df_metadata = aggregate_df(df, groupby=wellindex_col, func=np.unique)
    df_intensity.columns = df_intensity.columns.str.replace(value_col, name)
    df_plate = reshape_dataframe_to_plate(
        df_intensity, value_col=name, wellindex_col=wellindex_col
    )
    df_intensity = pd.merge(df_intensity, df_metadata, on=wellindex_col)
    df_intensity.attrs["name"] = df.attrs["name"] + "_{}-perWell".format(name)
    if save:
        save_dataframe_to_csv(df_intensity, path=output_directory)
    return df_plate
