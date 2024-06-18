import pandas as pd


def convert_wellindex_to_dataframe(df: pd.DataFrame, value_col):
    letters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
    ]
    df = df.copy()
    df["Row"] = df["WellIndex"].map(
        lambda x: letters[int(str(x).replace(".0", "")[:-3]) - 1]
    )
    df["Column"] = df["WellIndex"].map(lambda x: int(str(x).replace(".0", "")[-2:]))
    df_plate = pd.DataFrame(
        index=list(df.Row.sort_values().unique()),
        columns=df.Column.sort_values().unique(),
    )
    grps = df.groupby(["Row", "Column"])[value_col]
    for r, c in zip(df.Row, df.Column):
        df_plate.loc[r, c] = grps.get_group((r, c)).values.astype(int)[0]
    df_plate.fillna(0, inplace=True)
    return df_plate
