import pandas as pd
import h5py
import numpy as np
from ..tools import split_df_by_dtype


def save_dataframe_to_h5_file(filename: str, dataframe: pd.DataFrame, name: str = None):
    with h5py.File(filename, "a") as hf:
        data = dataframe.to_numpy(dtype="float64")
        index = dataframe.index.to_numpy(dtype="S")
        columns = dataframe.columns.to_numpy(dtype="S")
        hf.create_dataset("data", data=data, dtype=data.dtype)
        hf.create_dataset("index", data=index, dtype=index.dtype)
        hf.create_dataset("columns", data=columns, dtype=columns.dtype)
        if name is not None:
            hf.create_dataset(
                "name", data=np.array([name]), dtype="<U{}".format(len(name))
            )
        hf.close()


def save_mixed_dataframe_to_h5_file(
    filename: str, dataframe: pd.DataFrame, name: str = None
):
    data_num, data_cat = split_df_by_dtype(df=dataframe)
    with h5py.File(filename, "a") as hf:
        index_num = data_num.index.to_numpy(dtype="S")
        columns_num = data_num.columns.to_numpy(dtype="S")
        data_num = data_num.to_numpy(dtype="float64")

        index_cat = data_cat.index.to_numpy(dtype="S")
        columns_cat = data_cat.columns.to_numpy(dtype="S")
        data_cat = data_cat.to_numpy(dtype="S")

        hf.create_dataset("data_num", data=data_num, dtype=data_num.dtype)
        hf.create_dataset("index_num", data=index_num, dtype=index_num.dtype)
        hf.create_dataset("columns_num", data=columns_num, dtype=columns_num.dtype)

        hf.create_dataset("data_cat", data=data_num, dtype=data_num.dtype)
        hf.create_dataset("index_cat", data=index_cat, dtype=index_cat.dtype)
        hf.create_dataset("columns_cat", data=columns_cat, dtype=columns_cat.dtype)
        if name is not None:
            hf.create_dataset("name", data=np.array([name]), dtype="S")
        hf.close()
