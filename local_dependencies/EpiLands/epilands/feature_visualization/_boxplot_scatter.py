# import libraries
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from typing import List, Union

# relative imports
from ..generic_read_write import save_matplotlib_figure
from ..tools import join_iterable
from ..config import *
from ._plotting_utils import label_by_metadata


def plot_boxplot_scatter(
    df: pd.DataFrame,
    x_col: Union[str, list],
    y_col: str,
    title: str,
    figurename: str,
    boxplot_color_map: dict,
    graph_output_folder: str,
    color_by: str = None,
    scatter_color_map: dict = None,
    save_info: bool = False,
    order: list = None,
    **kwargs,
):
    fig, ax = plt.subplots()
    fig.set_facecolor("white")
    if isinstance(x_col, list):
        hue_label = join_iterable("scatter", x_col)
        df = label_by_metadata(
            df, label=hue_label, metadata_cols=color_by, mapping=scatter_color_map
        )
    else:
        hue_label = x_col

    if isinstance(y_col, list):
        if len(y_col) > 1:
            raise ValueError(
                f"plot_categorical_scatter: y_col must be a single value but {y_col} was given"
            )
        y_col = y_col[0]

    centroids = df.groupby(x_col)[y_col]
    sns.boxplot(
        x=x_col,
        y=y_col,
        hue=x_col,
        order=order,
        dodge=False,
        palette=boxplot_color_map,
        data=df,
        ax=ax,
    )
    sns.swarmplot(
        x=x_col,
        y=y_col,
        hue=color_by,
        order=order,
        marker="o",
        edgecolor="DarkSlategray",
        size=6,
        linewidth=1,
        alpha=0.9,
        palette=scatter_color_map,
        data=df,
        ax=ax,
        # zorder=-1
    )
    plt.legend(bbox_to_anchor=(1.5, 1))
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{x_col} vs {y_col} Scatter")
    plt.suptitle(title)
    centroid_data = pd.DataFrame.from_dict(
        {"centroid": centroids.mean(), "std": centroids.std()}
    )
    centroid_data.attrs["name"] = figurename + "_centroids"
    if save_info:
        save_matplotlib_figure(fig, path=graph_output_folder, figurename=figurename)
    return fig, ax


#        with pd.ExcelWriter(os.path.join(graph_output_folder, figurename+'.xlsx')) as writer:
#           for data, name in zip([df, centroid_data],['points','group_centroids']):
#              data.to_excel(writer, sheet_name=name)
# plt.show()
