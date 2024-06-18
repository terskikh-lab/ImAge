# Import libraries
from __future__ import annotations
import logging
import pandas as pd
import os
# relative imports
from ..tools import (
    TA_dropna,
    extract_timedelta_age_from_age_series,
    split_data_by_tissue,
)

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def prepare_data_for_cleaning(
    df: pd.DataFrame,
    sample_col: list = ["Sample"],
    condition_cols: list = ["Age", "Tissue", "Condition"],
):
    df_rd = TA_dropna(df)
    df_rd.loc[:, "AgeInput"] = df_rd["Age"]
    df_rd.loc[:, "Age"] = extract_timedelta_age_from_age_series(df_rd["Age"])
    df_tissue_dict = split_data_by_tissue(df_rd)
    return df_tissue_dict
