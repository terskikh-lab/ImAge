import re
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm
from typing import Union
import logging
from epilands.generic_read_write import (
    read_all_h5_outputs,
    find_all_files,
)
from epilands.tools import (
    extract_timedelta_age_from_age_series,
)
from ...generic_tags import (
    receives_nothing,
)
from ..tags import (
    receives_data,
    receives_platemap,
    outputs_data,
    outputs_model,
    outputs_platemap,
)

# sub_package_name = os.path.dirname(os.path.abspath(__file__)).split("/")[-2]
# logger = logging.getLogger(sub_package_name)
logger = logging.getLogger("processes")


@outputs_data
@receives_nothing
def load_h5_feature_data(
    feature_extraction_directory: str,
    search_pattern: Union[str, re.Pattern] = ".hdf5",
):
    return read_all_h5_outputs(
        feature_extraction_directory=feature_extraction_directory,
        raw_data_pattern=search_pattern,
    )


@outputs_data
@receives_nothing
def load_acapella_data(
    feature_extraction_directory: str,
    search_pattern: Union[str, re.Pattern] = "output.txt",
):
    files = find_all_files(path=feature_extraction_directory, pattern=search_pattern)
    data = [pd.read_csv(file, delimiter="\t") for file in files]
    df = pd.concat(data, axis=0).reset_index(drop=True)
    return df


@outputs_data
@receives_nothing
def load_csv_feature_data(
    loadPathRoot: str,
    filename: str,
):
    data = pd.read_csv(os.path.join(loadPathRoot, filename))
    return data


@outputs_platemap
@receives_nothing
def load_platemap(platemap_path: str, sheet_name: str = "platemap"):
    if platemap_path.endswith(".txt"):
        platemap = pd.read_csv(platemap_path, delimiter="   ")
    elif ".xls" in platemap_path:
        with pd.ExcelFile(platemap_path) as xls:
            platemap = pd.read_excel(xls, sheet_name)
    else:
        raise NotImplementedError(
            f"Expected platemap as txt or excel file, but {os.path.split(platemap_path)[-1]} was given"
        )
    for i, dt in enumerate(platemap.dtypes):
        if dt == "object":
            if platemap.iloc[:, i].nunique() / platemap.iloc[:, i].count() <= 0.5:
                platemap.iloc[:, i].astype("category", copy=False)
    return platemap


@outputs_data
@receives_platemap
@receives_data
def merge_platemap_with_data(
    data: pd.DataFrame, platemap: str, return_dask: bool = False
):
    check_shape = data.shape[0]

    platemap["BirthDeath"] = platemap["Age"]
    platemap["Age"] = extract_timedelta_age_from_age_series(platemap["Age"])
    if not data.index.is_unique:
        logger.warning("Data index is not unique. Dropping and resetting index")
        data.reset_index(drop=True, inplace=True)

    data = dd.from_pandas(data, chunksize=10000)
    platemap = dd.from_pandas(platemap, chunksize=10000)
    # data["WellIndex"] = data["WellIndex"]  # .map(lambda x: list(str(x))).map(lambda x: x[:1]+x[2:]).map(lambda x: ''.join(x)).astype(float).astype(int)
    data_merged = dd.merge(
        left=data,  # .loc[:, "WellIndex"],
        right=platemap,
        how="inner",
        on="WellIndex",
    )

    if return_dask == False:
        data_merged = data_merged.compute()
        if data_merged.shape[0] != check_shape:
            raise ValueError(
                f"ERROR: SOME CELLS WERE LOST IN MERGE\n\nshape before:{check_shape} | shape after:{data.shape[0]}.\n\nCHECK PLATEMAP AND RAW DATA INPUTS FOR MISSING ENTRIES"
            )
        for i in [1, 2, 3, 4]:
            if any(data_merged.columns.str.contains(f"ch{i}")) or any(
                data_merged.columns.str.contains(f"Ch{i}")
            ):
                ab = data_merged[f"Channel{i}PrimaryAntibody"].unique().tolist()
                for l in ["0", "0.0", np.nan, 0, 0.0]:
                    if l in ab:
                        logger.warning(
                            f"disallowed value {l} found in Channel{i}PrimaryAntibody, dropping rows"
                        )
                        ab.remove(l)
                        data_merged.drop(
                            data_merged.index[
                                (data_merged[f"Channel{i}PrimaryAntibody"] == l)
                            ],
                            axis="index",
                            inplace=True,
                        )

                        # data_merged = data_merged.loc[
                        #     ~(data_merged[f"Channel{i}PrimaryAntibody"] == l).values, :
                        # ]
                if len(ab) == 1:
                    ab = ab[0]
                else:
                    logger.warning(f"Ch{i} has more than one value: {ab}")
                # LETS IMPLEMENT THIS LATER
                # platemap.columns = platemap.columns.str.replace(channel, ab)
                # data_merged.columns = data_merged.columns.str.replace(f"ch{i}", ab)
                data_merged.rename(
                    columns=lambda c: c.replace(f"ch{i}", ab), inplace=True
                )
                data_merged.rename(
                    columns=lambda c: c.replace(f"Ch{i}", ab), inplace=True
                )

    return data_merged
