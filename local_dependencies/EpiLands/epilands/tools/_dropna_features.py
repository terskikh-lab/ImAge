import pandas as pd
import logging
import os
# General Cleaning Functions
sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

def TA_dropna(
    df: pd.DataFrame, axis: int or str = "columns", how: str = "all"
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
    df_nona = df.dropna(axis=axis, how=how)

    if axis in [0, "index"]:
        check = df.index
        check1 = df_nona.index
    elif axis in [1, "columns"]:
        check = df.columns
        check1 = df_nona.columns
    else:
        raise ValueError(
            "TA_dropna: Axis must be either (0, 'columns') for columns, or (1, 'index') for index)"
        )

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
