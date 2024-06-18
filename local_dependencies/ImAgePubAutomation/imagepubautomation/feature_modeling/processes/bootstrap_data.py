import pandas as pd
import re
from epilands import tools, feature_preprocessing
from ..tags import (
    receives_data,
    outputs_data,
)
from typing import List, Callable, Union
import logging

logger = logging.getLogger(name=__name__)


def group_and_drop_nonunqiue(df, group_by):
    # group the dataframe by the specified columns
    grouped = df.set_index(group_by).groupby(group_by)

    def _uniquecols(df):
        return df.apply(lambda c: len(set(c)) == 1)

    # uniquecols = grouped.apply(_uniquecols).min(axis=0)
    nonuniquecols = df.columns.drop(group_by)[
        grouped.nunique().max(axis=0) != 1
    ].to_list()

    def _returnfirstele(df):
        return df.iloc[0].drop(nonuniquecols)

    new_df = grouped.apply(_returnfirstele)
    return new_df


def bootstrap_samples(
    df,
    group_by,
    with_replacement: bool,
    metric,
    num_cells,
    num_bootstraps: int,
    seed: int,
    frac: float,
):
    def _bootstrap(df, b):
        return df.sample(
            n=num_cells, frac=frac, replace=with_replacement, random_state=seed + b
        ).apply(metric, axis=0)

    # group the dataframe by the specified columns
    grouped = df.groupby(group_by)

    group_sizes = grouped.size()
    if num_cells is not None:
        if any(group_sizes < num_cells):
            if with_replacement == False:
                for group in group_sizes.index[group_sizes < num_cells]:
                    print(f"Dropping {group}")
                    df = df.drop(grouped.get_group(group).index)
    # create a list of dictionaries to store the sampled data and the statistics
    samples = []
    for b in range(num_bootstraps):
        sample = grouped.apply(_bootstrap, b=b)
        sample.loc[:, "bootstrap"] = b
        samples.append(sample)

    # convert the list of dictionaries to a pandas dataframe
    result_df = pd.concat(samples, axis=0)

    return result_df, group_sizes


@outputs_data
@receives_data
def bootstrap_data(
    data: pd.DataFrame,
    subset: Union[str, re.Pattern],
    group_by: List[str],
    metric: Callable,
    num_cells: int,
    num_bootstraps: int,
    seed: int = None,
):
    data_num, data_cat = tools.split_df_by_dtype(df=data)
    data_num = data_num[tools.get_columns(data_num, pattern=subset)]
    data_num.loc[:, group_by] = data[group_by]
    data_cat_new = group_and_drop_nonunqiue(data_cat, group_by=group_by)
    data_num_new, group_sizes, _ = feature_preprocessing.bootstrap_df(
        df=data_num,
        group_by=group_by,
        metric=metric,
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )
    data_new = data_num_new.merge(
        data_cat_new,
        left_index=True,
        right_index=True,
    )
    data_new.reset_index(drop=False, inplace=True)
    return data_new, group_sizes
