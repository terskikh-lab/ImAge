import pandas as pd
import numpy as np


def split_df_by_dtype(df):
    # separate numerical and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return df[num_cols], df[cat_cols]
