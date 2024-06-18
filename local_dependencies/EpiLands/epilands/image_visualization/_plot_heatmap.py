from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple

from ..generic_read_write import save_matplotlib_figure


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    filename: str,
    output_directory: str,
    save: bool,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    plots a heatmap of the given dataframe using seaborn.

    df: dataframe to be plotted
    output_directory: str folder to save the graph to
    save: bool, if True, save the graph to the output_directory

    returns
    fig: matplotlib figure
    """
    fig, ax = plt.subplots(
        figsize=(df.shape[1], df.shape[0]),
        # dpi=DEFAULT_DPI
    )
    heatmap = sns.heatmap(
        df, ax=ax, square=True, annot=True, annot_kws={"fontsize": 8}, **kwargs
    )
    heatmap.set(title=title)
    plt.tick_params(axis="y", rotation=0)
    if save == True:
        save_matplotlib_figure(fig, path=output_directory, filename=filename, **kwargs)
    return fig, ax
