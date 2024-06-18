import pandas as pd
from ..generic_read_write import find_all_files


def read_qc_data(qc_output_directory: str) -> pd.DataFrame:
    qc_files = find_all_files(qc_output_directory, search_str="imageqc")
    qc_data = [pd.read_csv(qc_file) for qc_file in qc_files]
    qc_data1 = [
        df.dropna(
            axis="index",
            how="all",
            subset=df.columns[~df.columns.isin(["WellIndex", "row", "column", "FOV"])],
        )
        for df in qc_data
    ]
    df_imageQC = pd.concat(qc_data1).sort_values(
        by=["WellIndex", "row", "column", "FOV"]
    )
    return df_imageQC
