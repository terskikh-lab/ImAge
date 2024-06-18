# import libraries
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RELATIVE IMPORTS #
from ..generic_read_write import save_matplotlib_figure
from ..tools import get_kwargs


def plot_scatterplot(
    df,
    x_col,
    y_col,
    title: str,
    figurename: str,
    save_info,
    graph_output_folder,
    color_by: str or list = None,
    color_map: dict = None,
    shape_by: str or list = None,
    shape_map: dict = None,
    s: int = 50,
    alpha: float = 0.5,
    square_axes: bool = False,
    **kwargs
):
    save_matplotlib_figure_kwargs = get_kwargs(
        items=save_matplotlib_figure.__code__.co_varnames, **kwargs
    )
    scatterplot_kwargs = {
        i: kwargs[i] for i in kwargs if i not in save_matplotlib_figure_kwargs
    }

    fig, ax = plt.subplots(figsize=(10, 10))
    style_order = shape_map.keys() if shape_map is not None else None

    scatter = sns.scatterplot(
        x=x_col,
        y=y_col,
        hue=color_by,
        palette=color_map,
        legend="auto",
        style=shape_by,
        style_order=style_order,  # np.array(list(style_order)).astype(float),
        s=s,
        alpha=alpha,
        data=df,
        ax=ax,
        edgecolor="DarkSlateGray",
        **scatterplot_kwargs
    )
    scatter.set(title=title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    if square_axes:
        plt.axis("square")

    if save_info:
        save_matplotlib_figure(
            fig,
            path=graph_output_folder,
            figurename=figurename,
            **save_matplotlib_figure_kwargs
        )
    # plt.show()
    return fig, ax
