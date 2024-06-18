import pandas as pd
import logging
import os

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def reshape_dataframe_to_plate(
    df: pd.DataFrame,
    value_col: str,
    wellindex_col: str = "WellIndex",
    fov_col: str = None,
) -> pd.DataFrame:
    for i in [value_col, wellindex_col, fov_col]:
        if (i not in df) and (i is not None):
            logger.error(f"convert_wellindex_to_dataframe: column {i} not found in df")
            raise ValueError(
                f"convert_wellindex_to_dataframe: column {i} not found in df"
            )
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
    # df['Row'] = df[wellindex_col].map(lambda x: letters[int(str(x).replace('.0','')[:-3])-1])
    # df['Column'] = df[wellindex_col].map(lambda x: int(str(x).replace('.0','')[-2:]))
    if fov_col is None:
        df_plate = df.pivot(index="Row", columns="Column", values=value_col)
        df_plate.fillna(0, inplace=True)
        return df_plate
    if fov_col is not None:
        # if df[fov_col].astype(int).max() > 5:
        #     df[fov_col] = df[fov_col].astype(str).map(_fov_consecutive_to_rowcol_dict)
        # df['fov_row']=df[fov_col].astype(str).map(lambda x: x[0]).astype(int)
        # df['fov_col']=df[fov_col].astype(str).map(lambda x: x[1]).astype(int)
        # df_plate = df.pivot(index=['Row', 'fov_row'],
        #                     columns=['Column', 'fov_col'],
        #                     values=value_col)
        df_plate = df.pivot(
            index=["Row", "FieldOfView"], columns=["Column"], values=value_col
        )
        df_plate.fillna(0, inplace=True)
        return df_plate
