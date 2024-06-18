# Import libraries
from __future__ import annotations
import logging
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# relative imports
sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def zscore_df(df):
    return pd.DataFrame(
        data=StandardScaler().fit_transform(df), columns=df.columns, index=df.index
    )
