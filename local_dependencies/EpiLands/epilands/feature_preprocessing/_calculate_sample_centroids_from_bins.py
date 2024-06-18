# Import libraries
from __future__ import annotations
import pandas as pd
import numpy as np

# Relative imports
from ..generic_read_write import save_dataframe_to_csv
from .. import tools
from numpy.typing import ArrayLike

# from beartype import beartype


# @beartype
def calculate_sample_centroids_from_bins(
    df_distance: pd.DataFrame,
    group_col_list: list,
    sample_BioID: list = ["Sample"],
    group_col_name: str = "Group",
    calculate_sem: bool = True,
    calculate_group_centroids: bool = False,
    save_to: str = None,
    save_data: bool = False,
    **kwargs,
):
    """
    calculate_sample_centroids_from_bins
    Purpose:
    Re-create calculate_centroids_and_groupby_sample, but grouped by sample_col with centroids of the sample_col data calculated via groupby group_col_name
    Parameters:
    df_distance: a pandas dataframe that includes column labeled sample_col to distinguish biological replicates, group_col_name to distinguish experimental groups, and a column labeled 'Xprime' and a column labeled 'Yprime'
    group_col_list: a list of strings that represents the group labels in the Group column of the dataframe
    sample_col: a string or a list of strings that represents the name of the Sample column(s) in the dataframe
    calculate_sem: a bool which decides if standard error of the mean is calculated from the bin averages.
    calculate_group_centroids: a bool which decides if the group centroids are calculated. If false, the output will be sample averages only
    Returns:
    df_distances_groupby_sample: a pandas dataframe that includes a group_col_list of groups, a sample_col of samples, and a 'Xprime_ave' and a 'Yprime_ave' column that includes the average values of Xprime and Yprime for each group
    """
    # groupby the sample column, take the mean
    _groupby_sample = df_distance.groupby(sample_BioID)
    df_distances_groupby_sample = _groupby_sample.mean()

    # calculate the std of the bins -- this is mathematically equivalent to the standard error of the mean
    # MATH JUSTIFICATION
    # mu = single-cell population mean, N = single-cell population size, sigmap = single-cell population stdev,
    # mux = binave distribution mean, Nx = [Xbins] sample size (num of bins), sigmax = binaverage sample stdev
    # [Xbins] = bin averages, which are random variables selected from the original single-cell distribution with ***sample size 200***,
    # so they are distributed normally around mean mu with standard deviation sigmax
    # SEM = sigmap / sqrt(N) ... sigmax = sigmap / sqrt(Nx)
    # we want to find SEM in terms of sigmax and Nx.
    #
    # notice that N = single-cell population size ~= number of bins * bin size = Nx*200
    # N = Nx*200, therefore
    # SEM = sigmap/ sqrt(Nx*200)
    #
    # sigmax**2 = sigmap**2/Nx ---> sigmax**2/200 = sigmap**2/(Nx*200)
    # sqrt(sigmax**2/200) = sqrt(sigmap**2/(Nx*200)) ---> sigmax/sqrt(200) = sigmap/sqrt(Nx*200) = simgap/sqrt(N) = SEM
    # therefore,
    #
    # SEM = sigmax/sqrt(200)
    #
    if calculate_sem:
        sample_SEMS = _groupby_sample.std() / np.sqrt(200)
        # calculate stdev and orthogonal stdev
        df_distances_groupby_sample[sample_SEMS.columns + "_SEM"] = sample_SEMS

    # calculate centroids (means) per experimental group
    if calculate_group_centroids:
        _groupby_group = df_distance.groupby(group_col_list)
        print("Groups Generated:", _groupby_group.groups.keys())

        centroids = _groupby_group.mean()

        def SEM_propogation(sems):
            return np.sqrt((sems.pow(2)).sum())

        print(sample_SEMS)
        centroid_sems = sample_SEMS.groupby(group_col_list).apply(SEM_propogation)
        centroid_sems.columns = centroids.columns + "_SEM"
        centroids.loc[:, centroid_sems.columns] = centroid_sems

        create_centroid_samplename = lambda name: " ".join(name) + "_ave"
        centroids.loc[:, sample_BioID[0]] = centroids.index.map(
            create_centroid_samplename
        )
        centroids = centroids.groupby(sample_BioID).mean()

        df_distances_groupby_sample = df_distances_groupby_sample.append(centroids)

    df_distances_groupby_sample.attrs["name"] = df_distance.attrs["name"] + "_centroids"
    df_distances_groupby_sample.loc[:, group_col_name] = tools.pd.create_group_col(
        df_distances_groupby_sample, group_col_list, group_col_name, **kwargs
    )
    df_distances_groupby_sample.sort_values(by=group_col_name, inplace=True)
    print(df_distances_groupby_sample)
    if save_data:
        save_dataframe_to_csv(df_distances_groupby_sample, save_to, **kwargs)
    return df_distances_groupby_sample
