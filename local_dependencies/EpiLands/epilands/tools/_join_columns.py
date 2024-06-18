import pandas as pd
from typing import Sequence
from ..config import COLUMN_SEPARATOR


def join_columns(
    df: pd.DataFrame, columns: Sequence[str], delimeter: str = COLUMN_SEPARATOR
) -> pd.Series:
    cols_to_join = [df[i].astype(str) for i in columns]
    new_col = cols_to_join[0]
    for col in cols_to_join[1:]:
        new_col = new_col + delimeter + col
    return new_col.astype("category")
