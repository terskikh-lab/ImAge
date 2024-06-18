import pandas as pd
import warnings
from ..config import COLUMN_SEPARATOR


def create_group_col(
    df: pd.DataFrame, group_col_list: list, group_col_name: str = "Group"
):
    """
    -----------------
    Function to concatenate a set of columns
    -----------------
    Inputs:
        dataframe (mSamples, nFeatures) : a pandas dataframe containing the data
        group_col_list: a list of columns or multiintex that will be concatenated into a new column
        group_col_name: the name of the new column that will contain the concatenated group of columns
        **kwargs:
            sep: the separator that will be used for the concatenated columns
    Outputs:
        series: a pandas series (mSamples long) containing the group data
    -----------------
    """
    df_flat = df.reset_index()
    for i in group_col_list:
        if i not in df_flat.columns:
            raise ValueError(
                "create_group_col: {} was not found in the dataframe columns or index".format(
                    i
                )
            )
    if group_col_name in df_flat.columns:
        warnings.warn(
            "create_group_col: {} is already a column in the dataframe".format(
                group_col_name
            )
        )
    new_col_data = list(
        df_flat[group_col_list]
        .applymap(lambda x: str(x))
        .agg(COLUMN_SEPARATOR.join, axis=1)
    )
    grp_col = pd.Series(new_col_data, index=df.index, name=group_col_name)
    return grp_col
