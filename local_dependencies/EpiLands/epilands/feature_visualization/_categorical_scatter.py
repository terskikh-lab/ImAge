# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Union

# relative imports
from ..generic_read_write import save_matplotlib_figure
from ..tools import join_iterable


def plot_categorical_scatter(
    df: pd.DataFrame,
    x_col: Union[str, list],
    y_col: str,
    title: str,
    figurename: str,
    color_map: dict,
    graph_output_folder: str,
    save_info: bool = False,
    *,
    order: list = None,
    **kwargs,
):
    fig, ax = plt.subplots()
    fig.set_facecolor("white")
    if isinstance(x_col, list):
        if len(x_col) > 1:
            df[join_iterable(x_col)] = (
                df[x_col]
                .aggregate(lambda row: join_iterable(tuple(row)), axis=1)
                .dropna()
            )
        x_col = join_iterable(x_col)
    if isinstance(y_col, list):
        if len(y_col) > 1:
            raise ValueError(
                f"plot_categorical_scatter: y_col must be a single value but {y_col} was given"
            )
        y_col = y_col[0]
    centroids = df.groupby(x_col)[y_col]
    x = centroids.groups.keys()
    if not isinstance(order, type(None)):
        x = order
    for i, name in enumerate(x):
        ax.errorbar(
            x=[str(name)],
            y=centroids.get_group(name).mean(),
            yerr=centroids.get_group(name).std(),
            marker="o",
            ms=12,
            markeredgecolor="DarkSlategray",
            ecolor="black",
            capsize=12,
            mfc=color_map[join_iterable(name)],
        )
    sns.stripplot(
        x=x_col,
        y=y_col,
        hue=x_col,
        order=x,
        edgecolor="DarkSlategray",
        size=6,
        linewidth=1,
        jitter=0.1,
        alpha=0.9,  # 0.75,
        palette=color_map,
        data=df,
        ax=ax,
        # zorder=-1
    )
    plt.legend(bbox_to_anchor=(1.5, 1.05))
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{x_col} vs {y_col} Scatter")
    plt.suptitle(title)
    centroid_data = pd.DataFrame.from_dict(
        {"centroid": centroids.mean(), "std": centroids.std()}
    )
    centroid_data.attrs["name"] = figurename + "_centroids"
    if save_info:
        save_matplotlib_figure(fig, path=graph_output_folder, filename=figurename)
    return fig, ax


#        with pd.ExcelWriter(os.path.join(graph_output_folder, figurename+'.xlsx')) as writer:
#           for data, name in zip([df, centroid_data],['points','group_centroids']):
#              data.to_excel(writer, sheet_name=name)
# plt.show()
