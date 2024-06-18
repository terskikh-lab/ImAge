# import libraries
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# relative imports
from ..generic_read_write import save_matplotlib_figure
from ..tools import join_iterable


def plot_catplot_scatter(
    df: pd.DataFrame,
    x_col: list,
    y_col: str,
    kind: str,
    title: str,
    figurename: str,
    catplot_color_map: dict,
    graph_output_folder: str,
    color_by: str = None,
    centroid_by: str = None,
    scatter_color_map: dict = None,
    save_info: bool = False,
    order: list = None,
    **kwargs,
):
    grpby = []
    for i in [x_col, centroid_by, color_by]:
        if i is None:
            continue
        if isinstance(i, (list, tuple)):
            for ii in i:
                if ii in grpby:
                    continue
                grpby.append(ii)
        elif isinstance(i, (str, int, float, bool)):
            if i in grpby:
                continue
            grpby.append(i)
        else:
            raise ValueError("Cannot group data")

    if centroid_by is not None:
        grpby_centroids = df.groupby(grpby)
    else:
        grpby_centroids = df.groupby(grpby)

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

    g = sns.catplot(
        x=x_col,
        y=y_col,
        hue=x_col,
        order=order,
        dodge=False,
        palette=catplot_color_map,
        kind=kind,
        data=df,
    )
    sns.swarmplot(
        x=x_col,
        y=y_col,
        hue=color_by,
        order=order,
        marker="o",
        edgecolor="DarkSlategray",
        size=10,
        linewidth=1,
        alpha=0.9,
        palette=scatter_color_map,
        data=df
        if centroid_by is None
        else grpby_centroids.mean().dropna().reset_index(),
        ax=g.axes.flatten()[0],
        # zorder=-1
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1.5, 1))
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{x_col} vs {y_col} {kind} Scatter")
    plt.suptitle(title)
    if save_info:
        save_matplotlib_figure(
            fig=g.fig, path=graph_output_folder, figurename=figurename
        )
        centroid_data = pd.DataFrame.from_dict(
            {
                "centroid": grpby_centroids[y_col].mean().dropna(),
                "std": grpby_centroids[y_col].std().dropna(),
            }
        )
        centroid_data.attrs["name"] = figurename + "_centroids"
        with pd.ExcelWriter(
            os.path.join(graph_output_folder, figurename + ".xlsx")
        ) as writer:
            for data, name in zip(
                [df, centroid_data], ["points", f"{join_iterable(grpby)}_centroids"]
            ):
                data.to_excel(
                    writer, sheet_name=name
                )  ## Fix long centrid sheet name - maybe metadata sheet?
