# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# RELATIVE IMPORTS #
from ..generic_read_write import save_matplotlib_figure


def threshold_objects_histogram(
    series: pd.Series,
    high_threshold: float,
    low_threshold: float,
    range: tuple,
    title: str = None,
    filename: str = "object_size_thresholding",
    save_info: bool = False,
    graph_output_folder: str = None,
    set_xscale="linear",
    set_yscale="linear",
    cells_per_bin=10,
    xlabel: str = None,
    **kwargs
):
    fig, ax = plt.subplots()  # create a figure and axis
    series.plot.hist(
        bins=int(len(series) / cells_per_bin), title=title, range=range
    )  # plot the histogram of the threshold_metric_col
    if high_threshold is not np.inf:
        ax.axvline(
            x=high_threshold, color="r", linestyle="dashed", linewidth=2
        )  # plot a vertical line at the high threshold
    if low_threshold != 0:
        ax.axvline(
            x=low_threshold, color="r", linestyle="dashed", linewidth=2
        )  # plot a vertical line at the high threshold
    plt.xlabel(xlabel)
    ax.set_xscale(set_xscale)  # change the x-axis scale
    ax.set_yscale(set_yscale)  # change the y-axis scale
    if save_info == True:
        save_matplotlib_figure(fig, path=graph_output_folder, filename=filename)
    # plt.show() #show the figure
