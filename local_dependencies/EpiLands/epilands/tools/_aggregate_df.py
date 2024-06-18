import pandas as pd
from typing import Union, Callable
import numpy as np
from ._join_columns import join_columns
from ..config import COLUMN_SEPARATOR, NAME_SEPARATOR_


def aggregate_df(df: pd.DataFrame, groupby, func: Union[Callable, str]) -> pd.DataFrame:
    df = df.copy()
    # if isinstance(groupby, str) or (not isinstance(groupby, str) and len(groupby) == 1):
    #     if isinstance(groupby, str):
    #         newname = COLUMN_SEPARATOR.join([groupby, "groups"])
    #     else:
    #         newname = COLUMN_SEPARATOR.join([groupby[0], "groups"])
    #     df[newname] = df[groupby]
    #     df_groupby = df.groupby(newname, as_index=True)
    #     df_agg = df_groupby.agg(func)
    # newname = COLUMN_SEPARATOR.join(groupby)
    # newname = COLUMN_SEPARATOR.join([newname, "groups"])
    # df[newname] = join_columns(df, columns=groupby)
    df.set_index(groupby, inplace=True, drop=True)
    if func == "unique":
        df_agg = _aggregate_unique_cols2(
            df=df,
            group_by=groupby,
        )
        # df_groupby = df.groupby(newname, as_index=True)
        # aggregate_results = [
        #     i.aggregate(np.unique).apply(lambda l: l[0] if len(l) == 1 else np.NaN)
        #     for _, i in df_groupby
        # ]
        # df_agg = pd.concat(aggregate_results, axis=1).T.set_index(newname)
        # df_agg.dropna(axis=1, how="all", inplace=True)
    else:
        df_groupby = df.groupby(groupby)
        df_agg = df_groupby.agg(func)
    return df_agg


def _aggregate_unique_cols(df: pd.DataFrame):
    result = {}
    for col in df.columns:
        unique_data = df[col].unique()
        if len(unique_data) == 1:
            result[col] = unique_data[0]
        else:
            result[col] = np.NaN
    return pd.Series(result)


def _aggregate_unique_cols2(df, group_by):
    # Define a function to check if a column has only one unique value
    def has_single_value(col):
        return col.nunique() == 1

    # Group the DataFrame by the specified columns
    grouped = df.groupby(group_by)

    # Aggregate the data using the defined function to determine which columns to keep
    agg_data = grouped.apply(
        lambda x: x.loc[:, x.apply(has_single_value)].apply(pd.unique)
    ).droplevel(len(group_by))

    # Flatten the resulting DataFrame
    # flat_data = agg_data.reset_index(drop=True)
    # return flat_data

    return agg_data
