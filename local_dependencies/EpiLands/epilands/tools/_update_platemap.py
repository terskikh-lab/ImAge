import pandas as pd


def update_platemap(
    df: pd.DataFrame, platemap: pd.DataFrame
):  # OLD CODE FOR USE IF FIND FILEPATHS FUNCTION IS USED, file_folder_loc: str, search_str: str = 'platemap.txt'):
    """
    Reads in a platemap via a search for search_str from all files contained in file_folder_loc and its subfolders.
    This platemap overwrites all current overlapping platemap data, and the function returns this updated DataFrame.

    Inputs:
        df: pd.DataFrame containing the data
        search_str: str to find the platemap file
    Outputs:
        df_new: a pandas dataframe containing the updated platemap data
    """
    # if platemap is None:
    #     platemap_path = select_filepath_tk()
    #     platemap = pd.read_csv(str(platemap_path), sep="\t")
    cols_to_drop = platemap.columns[[(i in df.columns) for i in platemap.columns]]
    cols_to_drop = cols_to_drop.to_list()
    cols_to_drop.remove("WellIndex")
    df_new = df.drop(cols_to_drop, axis="columns")
    check_shape = df_new.shape[0]
    df_new = platemap.merge(
        df_new, how="inner", left_on="WellIndex", right_on="WellIndex"
    )
    df_new.attrs["name"] = df.attrs["name"]
    if df_new.shape[0] != check_shape:
        raise ValueError(
            f"Merge added or removed rows: {check_shape} rows before merge -> {df_new.shape[0]} rows after merge"
        )
    return df_new
