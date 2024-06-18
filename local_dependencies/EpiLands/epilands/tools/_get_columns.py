import pandas as pd
import re


def get_columns(df: pd.DataFrame, pattern: re.Pattern, regex: bool = True):
    return df.columns[df.columns.str.contains(pattern, regex=regex)]
