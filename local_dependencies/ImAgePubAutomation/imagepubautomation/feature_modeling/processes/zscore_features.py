import re
from typing import Union
import pandas as pd
import numpy as np
from epilands import tools, feature_preprocessing
from ..tags import (
    receives_data,
    outputs_data,
)


@outputs_data
@receives_data
def zscore_data(
    data: pd.DataFrame,
    group_by: list = None,
    subset: Union[list, np.ndarray, str, re.Pattern] = None,
    regex: bool = True,
    copy: bool = False,
):
    data = data.copy() if copy == True else data
    if isinstance(subset, (str, re.Pattern)):
        subset = tools.get_columns(data, pattern=subset, regex=regex)
    if subset is None:
        subset = data.columns
    if group_by is not None:
        df_zscore = data.groupby(group_by, as_index=False)[subset].apply(
            feature_preprocessing.zscore_df
        )
    else:
        df_zscore = feature_preprocessing.zscore_df(data[subset])
    return df_zscore
    # data.loc[df_zscore.index, df_zscore.columns] = df_zscore
    # return data
