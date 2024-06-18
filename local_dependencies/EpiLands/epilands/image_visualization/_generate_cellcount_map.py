from __future__ import annotations
import pandas as pd
import numpy as np

from ..tools import aggregate_df, reshape_dataframe_to_plate
from ..generic_read_write import save_dataframe_to_csv


def generate_cellcount_map(
    df: pd.DataFrame,
    wellindex_col: str = "WellIndex",
    sample_col: str = "Sample",
    output_directory: str = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    generates a cellcount map from a dataframe given the wellindex column and the sample column.

    df: pd.DataFrame

    save: bool tells the function whether to save the cellcount map to a file

    output_directory: str where to save the cellcount map

    wellindex_col: str the name of the column containing the wellindex

    sample_col: str the name of the column containing the sample names
    """
    df_metadata = aggregate_df(df, groupby=wellindex_col, func=np.unique)
    df_cellcounts = df.groupby([wellindex_col])[sample_col].count().reset_index()
    df_cellcounts.columns = df_cellcounts.columns.str.replace(sample_col, "CellCounts")
    df_cellcounts = pd.merge(df_cellcounts, df_metadata, on=wellindex_col)
    df_cellcounts.attrs["name"] = df.attrs["name"] + "_cellcounts_per_well"
    df_plate = reshape_dataframe_to_plate(
        df_cellcounts, value_col="CellCounts", wellindex_col=wellindex_col
    )
    if save == True:
        save_dataframe_to_csv(df_cellcounts, path=output_directory)
    return df_plate
