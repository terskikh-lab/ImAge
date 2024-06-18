import pandas as pd


def split_data_by_tissue(df: pd.DataFrame, tissue_col: str = "Tissue"):
    df_tissue_dict = {}  # initialize a dictionary of dataframes, one for each tissue
    for tissue in df[tissue_col].unique():  # iterate through the tissues
        print("Found tissue {} in dataframe".format(tissue))
        df_work = df[
            df[tissue_col] == tissue
        ]  # create a dataframe for the current tissue
        df_work.attrs["name"] = "_".join((df.attrs["name"], str(tissue)))
        df_tissue_dict[tissue] = df_work
    return df_tissue_dict
