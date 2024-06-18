from __future__ import annotations

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Relative imports
from ..tools import join_iterable

######## STILL IN PROGRESS


def generate_MIEL_distances(
    df_pdist: pd.DataFrame,
    reference_group_A: tuple,
    reference_group_B: tuple,
    group_col_list: list,
    save_to: str,
    save: bool = True,
    **kwargs,
):
    """
    generate_MIEL_distances

    Parameters
    ----------
    df_pdist : dataframe
        dataframe of distances.

    reference_group_A : tuple(string1, string2, ... stringN) where N = len(group_col_list)
        name of left reference group. Must be a unique value contained in the groups geated by df_pdist.groupby(group_col_list)

    reference_group_B : tuple(string1, string2, ... stringN) where N = len(group_col_list)
        name of right reference group. Must be a unique value contained in the groups geated by df_pdist.groupby(group_col_list)

    group_col_list : list
        list of column names that identify groups. Passed to df_pdist.groupby(group_col_list)

    Returns
    -------
    df_MIEL_distances : dataframe
        pandas dataframe of distances transformed via a linear transformation forming the MIEL/miBioAge axis
    """

    if isinstance(df_pdist.index, pd.MultiIndex):
        df_workinginfo = df_pdist.groupby(group_col_list)
    else:
        raise ValueError(
            "generate_MIEL_distances: df_pdist must have a multiindex in order to calculate MIEL distances (reference groups needed)"
        )
    # create the series X and Y which compose the X and Y coordinates of each sample in the new coordinate system (X = distance from reference_group_A, Y = distance from reference_group_B)
    # NOTE: this can be confusing... it helps to notice that the data with LOW X VALUES and HIGH Y VALUES (upper left) cooresponds to the REFERENCE_GROUP_A (FARTHER FROM B, CLOSER TO A)
    # likewise, HIGH X VALUES and LOW Y VALUES (lower right) cooresponds to the REFERENCE_GROUP_B (CLOSER TO B, FARTHER FROM A)

    X = df_workinginfo.get_group(reference_group_A).mean()
    X.attrs["name"] = "X (Distance From {})".format(join_iterable(reference_group_A))
    Y = df_workinginfo.get_group(reference_group_B).mean()
    Y.attrs["name"] = "Y (Distance From {})".format(join_iterable(reference_group_B))
    plt.figure()
    plt.scatter(x=X, y=Y)
    plt.title("X vs Y")
    plt.xlabel(X.attrs["name"])
    plt.ylabel(Y.attrs["name"])
    plt.axis("square")
