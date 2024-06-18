from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
import os

from ..tools import pixel_area_to_diameter


def threshold_objects_histogram(
    series: pd.Series,
    high_threshold: Union[int, float],
    low_threshold: Union[int, float],
    object_count: Union[int, float],
    display_quantile: Union[int, float],
    graph_output_folder: str,
    save: bool,
    title: str = None,
    set_xscale: str = "linear",
    set_yscale: str = "linear",
    cells_per_bin: int = 10,
    **kwargs,
) -> None:
    fig, ax = plt.subplots()  # create a figure and axis
    data_in_area = series.map(pixel_area_to_diameter)
    data_in_area.plot.hist(
        bins=int(object_count / cells_per_bin),
        title=title,
        range=[0, data_in_area.quantile(display_quantile)],
    )  # plot the histogram of the threshold_metric_col
    if high_threshold is not None:
        ax.axvline(
            x=pixel_area_to_diameter(high_threshold),
            color="r",
            linestyle="dashed",
            linewidth=2,
        )  # plot a vertical line at the high threshold
    if low_threshold is not None:
        ax.axvline(
            x=pixel_area_to_diameter(low_threshold),
            color="r",
            linestyle="dashed",
            linewidth=2,
        )  # plot a vertical line at the high threshold

    ax.set_xscale(set_xscale)  # change the x-axis scale
    ax.set_yscale(set_yscale)  # change the y-axis scale
    if save == True:
        fig.savefig(
            os.path.join(graph_output_folder, "Thresholding_histogram.jpg"), dpi=400
        )
    plt.show()  # show the figure
