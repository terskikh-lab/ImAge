import pandas as pd
import h5py


def read_dataframe_from_h5_file(filename: str) -> pd.DataFrame:
    with h5py.File(filename, "r") as hf:
        data = hf["data"][:]
        index = hf["index"][:]
        columns = hf["columns"][:].astype(str)
        df = pd.DataFrame(data, index=index, columns=columns)
        if "name" in hf.keys():
            df.attrs["name"] = str(hf["name"][:][0])
        hf.close()
    df.loc[:, "Path"] = filename
    return df


def read_mixed_dataframe_from_h5(
    filename: str, dataframe: pd.DataFrame, name: str = None
):
    """
    Reads a .h5 file given from ELTA extraction into a pandas DataFrame.txt
    \nRequires that the .h5 file has keys 'data', 'index', 'columns'
    \nif key 'name' exists an attribute '.attrs['name']' will be added to the DataFrame
    """
    with h5py.File(filename, "r") as hf:
        data_num = hf["data"][:]
        index_num = hf["index"][:]
        columns_num = hf["columns"][:].astype(str)
        df_num = pd.DataFrame(data_num, index=index_num, columns=columns_num)

        data_cat = hf["data"][:]
        index_cat = hf["index"][:]
        columns_cat = hf["columns"][:].astype(str)
        df_cat = pd.DataFrame(data_cat, index=index_cat, columns=columns_cat)

        df = pd.concat([df_cat, df_num])
        if "name" in hf.keys():
            df.attrs["name"] = str(hf["name"][:][0])
        hf.close()
    df.loc[:, "Path"] = filename
    return df
