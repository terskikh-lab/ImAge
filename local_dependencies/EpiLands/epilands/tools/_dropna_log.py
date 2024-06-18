import pandas as pd
from typing import Union
import logging
import os

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def dropna_log(
    df: pd.DataFrame, axis: Union[int, str] = "columns", how: str = "all", **kwargs
) -> pd.DataFrame:
    """
    This function uses the pandas .dropna() method to remove columns with NA values
    _______________________________________________________________________________
    Parameters:
        df: pd.DataFrame
            the input dataframe to drop columns from
        how: str = 'all' or 'any'
            Passed to df.dropna(how = how) to determine how to drop the NA columns
    """
    if axis not in ["columns", "index", 0, 1]:
        logger.error(
            "_dropna: 'axis' must be either (0, 'columns') for columns, or (1, 'index') for index)"
        )
        raise ValueError(
            "_dropna: 'axis' must be either (0, 'columns') for columns, or (1, 'index') for index)"
        )
    df_nona = df.dropna(axis=axis, how=how)

    if axis in [0, "index"]:
        check = df.index
        check1 = df_nona.index
    elif axis in [1, "columns"]:
        check = df.columns
        check1 = df_nona.columns
    for i in check:
        if i not in check1:
            logger.warning(
                "Dropped {} from {} because it contained {} NA values".format(
                    i, axis, how
                )
            )
    try:
        df_nona.attrs["name"] = df.attrs["name"]
    finally:
        return df_nona
