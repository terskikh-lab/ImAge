import pandas as pd
import re
import os
from typing import Union
import logging
from beartype import beartype

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

_row_lettertonum_dict = {
    "A": "1",
    "B": "2",
    "C": "3",
    "D": "4",
    "E": "5",
    "F": "6",
    "G": "7",
    "H": "8",
    "I": "9",
    "J": "10",
    "K": "11",
    "L": "12",
    "M": "13",
    "N": "14",
    "O": "15",
    "P": "16",
}


def parse_imagelike_filename_metadata(
    files: pd.Series,
    pattern: re.Pattern,
    channelIndex: int,
    rowIndex: int,
    colIndex: int,
    zIndex: int,
    FOVIndex: Union[tuple, int],
    tIndex: int,
) -> pd.DataFrame:
    file_information = pd.DataFrame(files, columns=["file_path"])
    get_tail = lambda s: os.path.split(s)[1]
    file_information["filename"] = file_information["file_path"].apply(get_tail)
    match = file_information["filename"].str.extract(pattern, expand=False)
    if channelIndex is not None:
        file_information["channel"] = match[channelIndex].astype(str)
    if rowIndex is not None:
        if (
            match[rowIndex].astype(str).str.contains(pat="[a-zA-Z]", regex=True).max()
            == 0
        ):
            file_information["row"] = match[rowIndex].astype(int)
        else:
            logger.warning("Alphabetical Rows Detected")
            check = match[rowIndex].isna().sum()
            file_information["row"] = (
                match[rowIndex].astype(str).map(_row_lettertonum_dict).astype(int)
            )
            if file_information["row"].isna().sum() != check:
                print(match[rowIndex])
                raise ValueError(
                    f"{match[rowIndex].isna().sum()-check} Alphabetical rows detected could not be converted to ints"
                )
    if colIndex is not None:
        file_information["column"] = match[colIndex].astype(int)
    if zIndex is not None:
        file_information["zstack"] = match[zIndex].astype(int)
    if FOVIndex is not None:
        if isinstance(FOVIndex, (list, tuple)):  # INTEGRATE THIS LATER ON
            FOVrow = match[FOVIndex[0]].map(lambda x: str(int(x)))
            FOVcol = match[FOVIndex[1]].map(lambda x: str(int(x)))
            file_information["FOV"] = (FOVrow + FOVcol).astype(int)
        else:
            file_information["FOV"] = match[FOVIndex].astype(int)
    if tIndex is not None:
        file_information["time"] = match[tIndex].astype(int)
    # Calculate the WellIndex for each image. Use this to calculate Number of Wells later
    file_information["WellIndex"] = (
        file_information["row"].astype(str)
        + file_information["column"]
        .astype(str)
        .map(lambda s: "00" + s if len(s) == 1 else "0" + s)
    ).astype(int)
    return file_information


@beartype
def _join_wellindex(rows: pd.Series, columns: pd.Series):
    rowlen = rows.map(lambda x: len(str(x)))
    collen = columns.map(lambda x: len(str(x)))
    if max(rowlen) > 2:
        raise ValueError("Detected more than 99 rows in the plate")
    if max(collen) > 2:
        raise ValueError("Detected more than 99 columns in the plate")
    rows = rows.astype(str)
    rows.loc[rowlen == 1] = rows[rowlen == 1].map(lambda i: "0" + str(i))
    columns = columns.astype(str)
    columns.loc[collen == 1] = columns[collen == 1].map(lambda i: "0" + str(i))
    return (rows + columns).astype(int)
