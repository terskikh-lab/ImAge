# Import libraries
from __future__ import annotations
from typing import List
import numpy as np
from tqdm import tqdm

# relative imports


def bin_df(df, group_by, num_cells: int = 200):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe containing single cell data that has been preprocessed in some way
    num_cells : int
        number of cells to sample per bin
    Returns
    -------
    df_work : pandas dataframe
        dataframe containing single cell data that has been binned
    """
    # logg.info(f"Began binning single cell data into bins of {num_cells} cells")
    df_work = df.copy()
    df_work["bin"] = np.NaN
    groupby_sample = df_work.groupby(group_by, as_index=False)
    for sample in tqdm(
        groupby_sample.groups, f"Progress binning cells into bins of {num_cells}:"
    ):
        sample_cells = groupby_sample.get_group(sample).index
        bins = int(np.floor(len(sample_cells) / num_cells))
        choice = np.random.choice(sample_cells, size=(bins, num_cells), replace=False)
        for b in range(choice.shape[0]):
            df_work.loc[choice[b], "bin"] = np.int16(b + 1)
        # logg.info(tools.utils.join_iterable(sample)
        #           +"\nBinning completed:\n"
        #           +f"{bins} bins created for {len(sample_cells)} cells. "
        #           +f"A total of {len(sample_cells )-bins*num_cells} "
        #           +"cells were lost in binning")
    df_work.drop(df_work.index[df_work["bin"].isna()], axis="index", inplace=True)
    # logg.info(f'SUCCESS: binned single cell data into bins of {num_cells}'
    #           +"cells, losing {} cells in total ({}%)".format(len(df.index)-len(df_work.index),
    #                                                           (len(df.index)-len(df_work.index))/len(df.index)*100))
    return df_work
