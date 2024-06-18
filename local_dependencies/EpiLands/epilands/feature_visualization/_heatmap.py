# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RELATIVE IMPORTS #
from ..generic_read_write import save_matplotlib_figure
from ..tools import join_iterable


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    figurename: str,
    graph_output_folder: str,
    save_info: bool,
    **kwargs
):
    """
    plots a heatmap of the given dataframe using seaborn.

    df: dataframe to be plotted
    graph_output_folder: str folder to save the graph to
    save_info: bool, if True, save the graph to the graph_output_folder

    returns
    fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap = sns.heatmap(
        df,
        annot=True if len(df) < 100 else False,
        fmt="d" if df.dtypes.unique()[0] != np.dtype("float") else ".2f",
        ax=ax,
        **kwargs
    )
    heatmap.set(title=title)
    plt.tick_params(axis="y", rotation=0)
    if save_info == True:
        save_matplotlib_figure(
            fig, path=graph_output_folder, figurename=figurename, **kwargs
        )
    return fig


import plotly.express as px
import os


# import dash_bio
def plotly_heatmap(
    df: pd.DataFrame,
    title: str,
    figurename: str,
    graph_output_folder: str,
    save_info: bool,
    **kwargs
):
    """
    plots a heatmap of the given dataframe using seaborn.

    df: dataframe to be plotted
    graph_output_folder: str folder to save the graph to
    save_info: bool, if True, save the graph to the graph_output_folder

    returns
    fig: matplotlib figure
    """

    # x_labels = np.fromiter((join_iterable(xi) for xi in df.columns), df.columns.dtype)
    heatmap = px.imshow(
        df,
        x=list((join_iterable(xi) for xi in df.columns)),
        y=list((join_iterable(xi) for xi in df.columns)),
        title=title,
        **kwargs
    )
    if save_info == True:
        heatmap.write_html(os.path.join(graph_output_folder, figurename + ".html"))
