import pandas as pd


def extract_multiindex_values(df: pd.DataFrame, axis: str = "rows"):
    df_work = df.copy()
    valid = ["rows", "columns", "both"]
    if axis not in valid:
        raise ValueError("collapse_multiindex: status must be one of %r." % valid)

    if axis in ["rows", "both"]:
        df_work.reset_index(inplace=True)

    if axis in ["columns", "both"]:
        mi = [df.columns.get_level_values(i) for i in range(len(df.columns.levels))]
        for i in mi:
            df_work.loc[i.name, :] = i
    df_work.attrs["name"] = df.attrs["name"]
    return df_work
