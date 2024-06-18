from __future__ import annotations
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import logging
import os

from ..generic_read_write import save_matplotlib_figure
from ..tools import reshape_dataframe_to_plate
from ._plot_heatmap import plot_heatmap
from multiprocesstools import MultiProcessHelper
import os

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def plot_qc_heatmaps(
    df_imageQC: pd.DataFrame, output_directory: str, save: bool = True
) -> None:
    # plot the image QC data
    multi_process_helper = MultiProcessHelper(
        name="plot_qc_heatmaps",
        working_directory=output_directory,
        loggers=[__name__],
    )

    if "FieldOfView" in df_imageQC.columns:
        num_fovs = df_imageQC["FieldOfView"].unique().shape[0]
    else:
        num_fovs = 1

    for col in df_imageQC.columns:
        if col not in ["WellIndex", "FieldOfView", "Column", "Row"]:
            fig_name = f"{col}_heatmap"
            file_not_in_use = multi_process_helper.create_temp_file(
                final_file_name=fig_name + ".pdf",
                temp_file_name=fig_name + ".tmp",
                path=output_directory,
            )
            if file_not_in_use == True:
                pass
            elif file_not_in_use == False:
                logger.debug(f"File {fig_name} already exists, skipping...")
                continue
            df_plate = reshape_dataframe_to_plate(
                df_imageQC.reset_index(),
                value_col=col,
                wellindex_col="WellIndex",
                fov_col="FieldOfView",  # FUTURE: add grid capabilities, must modify h/v-lines below
            )
            fig, ax = plot_heatmap(
                df=df_plate,
                title=f"{col}",
                filename=fig_name,
                output_directory=output_directory,
                save=False,
            )
            ax.vlines(
                np.arange(start=1, stop=df_plate.shape[1], step=1),
                *ax.get_ylim(),
                linewidth=4,
                color="black",
                label="TEST",
            )
            ax.hlines(
                np.arange(start=num_fovs, stop=df_plate.shape[0], step=num_fovs),
                *ax.get_xlim(),
                linewidth=4,
                color="black",
                label="TEST",
            )
            if save == True:
                save_matplotlib_figure(fig, path=output_directory, filename=fig_name)
            plt.close()
            multi_process_helper.delete_tempfile(
                tempfile=os.path.join(output_directory, fig_name + ".tmp")
            )
    multi_process_helper.cleanup()
