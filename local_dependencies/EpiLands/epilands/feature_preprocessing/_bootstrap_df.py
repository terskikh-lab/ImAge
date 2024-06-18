# Import libraries
from __future__ import annotations
import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple, Union
from numbers import Number
import time
import dask.dataframe as dd
import dask.array as da

# from tqdm.notebook import tqdm, trange

# relative imports

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def bootstrap_df(
    df: pd.DataFrame,
    group_by: list,
    metric: Callable,
    num_cells: Union[int, str],
    num_bootstraps: int,
    seed: int = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe containing single cell data that has been preprocessed in some way
    groupby : str or list
        column name or list of column names to group by
    metric : callable
        function to apply to each group of cells
    num_bootstraps : int
        number of bootstraps to perform
    num_cells : int
        number of cells to sample per bootstrap
    Returns
    -------
    bootstrap_samples : pandas dataframe
        dataframe containing bootstrap samples
    group_sizes : dict
        dictionary of original group sizes (N)
    seed : int
        random state used in bootstrap sampling
    """
    if seed == None:
        seed = np.random.randint(low=0, high=2**16, size=None)
    elif not isinstance(seed, Number):
        raise TypeError(f"seed {seed} is not a number")
    if isinstance(group_by, str) or len(group_by) == 1:
        df_groups = df.set_index(group_by).drop(group_by, axis=1)
    else:
        df_groups = df.set_index(df[group_by].astype(str).apply("|".join, axis=1)).drop(
            group_by, axis=1
        )
        group_by = "|".join(group_by)
        df_groups.index.name = group_by
    df = None

    group_sizes = df_groups.index.value_counts().sort_index()
    # num_bootstraps = int(
    #     min(group_sizes) // num_cells if num_bootstraps is None else num_bootstraps
    # )
    # if num_bootstraps <= 1:
    #     raise ValueError(f"Not enough cells for bootstrapping")
    if isinstance(num_cells, int):
        if any(group_sizes < num_cells):
            for name, grpsize in group_sizes.items():
                print(name, grpsize)
                if grpsize < num_cells:
                    # logger.warning(
                    #     f"Dropping group, not enough cells for bootstrapping: {grp} ({grpsize}) cells"
                    # )
                    # df_groups.drop(name, inplace=True)
                    logger.error(
                        f"Not enough cells for bootstrapping {num_cells} cells in group: {name} ({grpsize}) cells"
                    )
            raise ValueError(f"Not enough cells for bootstrapping")
        logger.debug(
            f"Began calculating {metric} bootstrapping {num_bootstraps} samples of {num_cells} cells from {group_by}"
        )

    rng = np.random.default_rng(seed=seed)

    if metric == "mean":
        _bootdf = lambda df: df.mean(axis=0)
    elif metric == "std":
        _bootdf = lambda df: df.std(axis=0)
    elif isinstance(metric, Callable):
        _bootdf = lambda df: df.apply(metric, axis=0)

    if num_cells == 1:
        logger.warn(f"num_cells = 1, so returning value rather than {metric}")
        bootstrap_samples = []
        for name, size in group_sizes.items():
            # start = time.perf_counter()
            dfgrp = df_groups.loc[name, :]
            sample = rng.choice(size, size=num_cells * num_bootstraps, replace=True)
            idx = [name] * num_bootstraps
            bootstrap_data = dfgrp.iloc[sample, :]
            bootstrap_data.index = idx
            bootstrap_data.loc["bootstrap"] = 1
            bootstrap_samples.append(
                pd.DataFrame(bootstrap_data, index=[name] * num_bootstraps)
            )
            # stop = time.perf_counter()
            # print(stop - start)

    elif num_cells == "original":
        bootstrap_samples = []
        for name, size in group_sizes.items():
            # start = time.perf_counter()
            dfgrp = df_groups.loc[name, :]
            bootstrap_data = []
            sample = rng.choice(size, size=(num_bootstraps, size), replace=True)
            for i, bsample in enumerate(sample):
                df = _bootdf(dfgrp.iloc[bsample, :])
                df.loc["bootstrap"] = i + 1
                bootstrap_data.append(df)
            bootstrap_samples.append(
                pd.DataFrame(bootstrap_data, index=[name] * num_bootstraps)
            )
            # stop = time.perf_counter()
            # print(stop - start)
    else:
        bootstrap_samples = []
        for name, size in group_sizes.items():
            # start = time.perf_counter()
            dfgrp = df_groups.loc[name, :]
            bootstrap_data = []
            # CHANGE SIZE TO 2D ARRAY (num_cells, num_bootstraps) ~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # sample = rng.choice(size, size=num_bootstraps*num_cells, replace=True)
            sample = rng.choice(size, size=(num_bootstraps, num_cells), replace=True)
            for i, bsample in enumerate(sample):
                # bsample = sample[b * num_cells : b * num_cells + num_cells]
                df = _bootdf(dfgrp.iloc[bsample, :])
                df.loc["bootstrap"] = i + 1
                bootstrap_data.append(df)
            bootstrap_samples.append(
                pd.DataFrame(bootstrap_data, index=[name] * num_bootstraps)
            )
            # stop = time.perf_counter()
            # print(stop - start)

    bootstrap_samples = pd.concat(
        bootstrap_samples
    )  # concatenate the bootstrap samples

    group_by = group_by.split("|")
    original_cols = bootstrap_samples.index.str.extract(
        "\|".join(["([A-Za-z0-9-_]+)"] * len(group_by))
    ).rename({i: col for i, col in enumerate(group_by)}, axis=1)
    bootstrap_samples.reset_index(inplace=True, drop=True)
    bootstrap_samples[group_by] = original_cols
    bootstrap_samples.set_index(group_by, inplace=True)
    logger.debug(f"Bootstrapping completed")
    return bootstrap_samples, group_sizes, seed


# VERSION 2

# def bootstrap_df(
#     df: pd.DataFrame,
#     group_by: list,
#     with_replacement: bool,
#     metric: Callable,
#     num_bootstraps: int,
#     num_cells: int,
#     frac: float = None,
#     seed: int = np.random.seed(),
# ) -> Tuple[pd.DataFrame, pd.Series]:
#     """
#     Parameters
#     ----------
#     df : pandas dataframe
#         dataframe containing single cell data that has been preprocessed in some way
#     groupby : str or list
#         column name or list of column names to group by
#     metric : callable
#         function to apply to each group of cells
#     num_bootstraps : int
#         number of bootstraps to perform
#     num_cells : int
#         number of cells to sample per bootstrap
#     Returns
#     -------
#     bootstrap_samples : pandas dataframe
#         dataframe containing bootstrap samples
#     group_sizes : dict
#         dictionary of original group sizes (N)
#     seed : int
#         random state used in bootstrap sampling
#     """
#     if seed == None:
#         seed = np.random.randint(low=0, high=2**16, size=None)
#     elif not isinstance(seed, Number):
#         raise TypeError(f"seed {seed} is not a number")
#     logger.debug(
#         f"Began calculating {metric} bootstrapping {num_bootstraps} samples of {num_cells} cells from {group_by}"
#     )
#     if isinstance(group_by, str) or len(group_by) == 1:
#         df_groups = df.set_index(group_by).drop(group_by, axis=1)
#     else:
#         df_groups = df.set_index(df[group_by].apply("_".join, axis=1)).drop(
#             group_by, axis=1
#         )
#         group_by = "_".join(group_by)
#         df_groups.index.name = group_by

#     group_sizes = df_groups.index.value_counts().sort_index()
#     if any(group_sizes < num_cells):
#         for name, grpsize in group_sizes.items():
#             print(name, grpsize)
#             if grpsize < num_cells:
#                 # logger.warning(
#                 #     f"Dropping group, not enough cells for bootstrapping: {grp} ({grpsize}) cells"
#                 # )
#                 # df_groups.drop(name, inplace=True)
#                 raise ValueError(
#                     f"Not enough cells for bootstrapping in group: {grp} ({grpsize}) cells"
#                 )

#     rng = np.random.default_rng(seed=seed)

#     if metric == "mean":
#         _bootdf = lambda df: df.mean(axis=0)
#     elif metric == "std":
#         _bootdf = lambda df: df.std(axis=0)
#     elif isinstance(metric, Callable):
#         _bootdf = lambda df: df.apply(metric, axis=0)

#     bootstrap_samples = []
#     for name, size in group_sizes.items():
#         dfgrp = df_groups.loc[name, :]
#         bootstrap_data = []
#         for b in range(num_bootstraps):
#             sample = rng.choice(size, size=num_cells, replace=with_replacement)
#             bootstrap_data.append(_bootdf(dfgrp.iloc[sample, :]))
#         bootstrap_samples.append(
#             pd.DataFrame(bootstrap_data, index=[name] * num_bootstraps)
#         )

#     bootstrap_samples = pd.concat(
#         bootstrap_samples
#     )  # concatenate the bootstrap samples

#     group_by = group_by.split("_")
#     original_cols = bootstrap_samples.index.str.extract(
#         "_".join(["([A-Za-z0-9-]+)"] * len(group_by))
#     ).rename({i: col for i, col in enumerate(group_by)}, axis=1)
#     bootstrap_samples.reset_index(inplace=True, drop=True)
#     bootstrap_samples[group_by] = original_cols
#     bootstrap_samples.set_index(group_by, inplace=True)
#     logger.debug(f"Bootstrapping completed")
#     return bootstrap_samples, group_sizes, seed


# ORIGINAL
# for b in range(num_bootstraps):  # for each bootstrap
#     bootstrap_result = ddf_groups.apply(
#         _bootdf,
#         meta=pd.DataFrame(columns=df_groups.columns, dtype=df_groups.dtypes[1]),
#     )
#     bootstrap_samples.append(
#         bootstrap_result
#     )  # add the bootstrap sample to the list

# bootstrap_result = ddf_groups.apply(
#     _bootstrap,  # apply the bootstrap function to each group
#     num_cells=num_cells,
#     frac=frac,
#     replace=with_replacement,
#     metric=metric,
#     seed=seed + b,
#     meta=pd.Series(index=df_groups.columns, dtype=df_groups.dtypes[1]),
# ).compute()  # add bootstrap number to seed for initialization
# # bootstrap_result.dropna(
# #     axis=0, how="all", inplace=True  # drop any rows that contain all NaN
# # )
# # if bootstrap_result.index.name in bootstrap_result.columns:
# #     bootstrap_result.drop(
# #         bootstrap_result.index.name, axis="columns", inplace=True
# #     )
# # bootstrap_result.reset_index(inplace=True)  # reset the index
# # bootstrap_result["Bootstrap"] = int(b)  # add a column with the bootstrap number
# bootstrap_samples.append(
#     bootstrap_result
# )  # add the bootstrap sample to the list

# df_groups = df.set_index(group_by, drop=True)
# if with_replacement == False:
#     for grp, _dat in df_groups:
#         max_bootstraps = _dat.shape[0] // num_cells
#         # num_bootstraps = num_bootstraps * 4 // 3
#         # if num_bootstraps > max_bootstraps:
#         # raise ValueError(
#         #     f"Not enough cells for {num_bootstraps} bootstraps to be in training set for group {grp}"
#         # )
# # if with_replacement == True:
# #     for grp, _dat in df_groups:
# #         # if _dat.shape[0] < num_cells:
# #         if _dat.shape[0] < num_bootstraps:
# #             raise ValueError(f"Not enough cells for bootstrapping in group {grp}")

# bootstrap_samples = []  # create a list to store the bootstrap samples
# # group_sizes = pd.Series(
# #         {str(group): data.shape[0] for group, data in df_groups}, name="group_sizes"
# #     )
# # if any(
# #     np.array(list(group_sizes.values())) < num_cells
# # ):  # if any group is too small to sample, raise an error and say why
# #     for group, size in group_sizes.items():
# #         if size < num_cells:
# #             logg.error(
# #                 f"Group {group} has {size} cells, which is less than {num_cells} cells per bootstrap."
# #             )
# #             raise ValueError(
# #                 f"Group {group} has {size} cells, which is less than {num_cells} cells per bootstrap."
# #             )

# # for b in tqdm(range(num_bootstraps)):  # for each bootstrap
# for b in range(num_bootstraps):  # for each bootstrap
#     bootstrap_result = df_groups.apply(
#         _bootstrap,  # apply the bootstrap function to each group
#         num_cells=num_cells,
#         frac=frac,
#         replace=with_replacement,
#         metric=metric,
#         seed=seed + b,
#     )  # add bootstrap number to seed for initialization
#     # bootstrap_result.dropna(
#     #     axis=0, how="all", inplace=True  # drop any rows that contain all NaN
#     # )
#     # if bootstrap_result.index.name in bootstrap_result.columns:
#     #     bootstrap_result.drop(
#     #         bootstrap_result.index.name, axis="columns", inplace=True
#     #     )
#     # bootstrap_result.reset_index(inplace=True)  # reset the index
#     bootstrap_result["Bootstrap"] = int(b)  # add a column with the bootstrap number
#     bootstrap_samples.append(
#         bootstrap_result
#     )  # add the bootstrap sample to the list
# bootstrap_samples = pd.concat(
#     bootstrap_samples
# )  # concatenate the bootstrap samples
# # bootstrap_samples.reset_index(inplace=True, drop=False)
# # logg.info(
# #     f"SUCCESS: bootstrapped {num_bootstraps} samples of {num_cells}"
# #     + f" for {len(df_groups.groups)} groups identified in {group_by}"
# # )

# return bootstrap_samples, group_sizes, seed


# def _bootstrap(
#     df: pd.DataFrame,
#     num_cells: int,
#     replace: bool,
#     metric: Callable,
#     seed: int,
#     frac: float = None,
# ):
#     """
#     df : pandas dataframe
#         dataframe containing a group of cells
#     num_cells : int
#         number of cells to sample per bootstrap
#     metric : callable
#         function to apply to each group of cells
#     Returns
#     -------
#     bootstrap_result : pandas dataframe
#         dataframe containing the bootstrap sample
#     """
#     try:
#         bootstrap_sample = df.sample(
#             n=num_cells, frac=frac, replace=replace, random_state=seed
#         )
#         bootstrap_result = bootstrap_sample.apply(metric, axis=0)
#     except ValueError as e:
#         if df.shape[0] < num_cells:
#             print(f"Not enough cells to bootstrap, returning NaN")
#             logger.error(e)
#             logger.warning(f"Not enough cells to bootstrap, returning NaN")
#             bootstrap_result = df.apply(lambda s: np.NaN, axis=0)
#         else:
#             raise e
#     # bootstrap_result["original_count"] = int(df.shape[0])
#     bootstrap_result.name = "bootstrap_result"
#     return bootstrap_result
