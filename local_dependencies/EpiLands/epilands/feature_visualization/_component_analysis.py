# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List

# relative imports
from ..generic_read_write import save_matplotlib_figure, save_dataframe_to_csv
from ..tools import join_iterable
from ._plotting_utils import make_space_above


def plot_component_loading_distribution(
    df_components: pd.DataFrame,
    sort: bool,
    channels: list,
    channel_color_map: dict,
    title: str,
    figurename: str,
    save_info: bool,
    graph_output_folder: str,
    **kwargs,
):
    plt.Figure()
    fig, ax = plt.subplots(len(df_components.columns), 2)
    fig.set_size_inches(18.5, 10.5)
    fig.set_facecolor("white")
    fig.suptitle(title, size=24)
    for i, component in enumerate(df_components.columns.to_list()):
        ax_tmp = ax[i][0] if len(df_components.columns) > 1 else ax[0]
        # data = df_components.loc[:,component].map(abs).sort_values(ascending = False)
        data = (
            df_components.loc[:, component].sort_values(ascending=False)
            if sort == True
            else df_components.loc[:, component]
        )
        ch_list = data.index.str.split("_")
        x = np.linspace(0, len(data), num=len(data))
        ax_tmp.bar(
            x, data, color=ch_list.map(lambda x: channel_color_map[x[0]]), width=1
        )
        ax_tmp.set_xticks(
            x, data.index.tolist(), rotation="vertical", fontsize=0.1 * 740 / len(data)
        )
        ax_tmp.set_title(f"{component} Feature Wieghts")
        labels = list(channel_color_map.keys())
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=channel_color_map[label])
            for label in labels
        ]
        ax_tmp.legend(handles, labels)
        ax_tmp.set_ylabel("Feature Weight", size=14)
        plot_pie_on_ax(
            ax=ax[i][1] if len(df_components.columns) > 1 else ax[1],
            component=component,
            df_components=df_components,
            channels=channels,
            channel_color_map=channel_color_map,
        )
    plt.tight_layout()
    # plt.legend()
    if save_info:
        save_matplotlib_figure(
            fig, path=graph_output_folder, figurename=figurename, **kwargs
        )
    # plt.show()


def plot_pie_on_ax(ax, component, df_components, channels, channel_color_map, **kwargs):
    def func(pct, allvals):
        absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
        return "{:.1f}%".format(pct, absolute)

    # Calculate vector magnitude
    tot_mag = np.linalg.norm(df_components.loc[:, component])
    # Calculate the ratio of each component to the the total vector magnitude
    data = [
        np.linalg.norm(
            df_components.loc[df_components.index.str.contains(ch), component]
        )
        / tot_mag
        for ch in channels
    ]
    wedges, texts, autotexts = ax.pie(
        data,
        autopct=lambda pct: func(pct, data),
        labels=channels,
        colors=[channel_color_map[ch] for ch in channels],
    )
    ax.set_facecolor("white")
    ax.set_title(component)


def plot_component_pie_and_scatter(
    df_component_analysis: pd.DataFrame,
    df_components: pd.DataFrame,
    title: str,
    figurename: str,
    color_by: List[str],
    color_map: dict,
    channels: list,
    channel_color_map: dict,
    graph_output_folder: str,
    save_info: bool,
    shape_by: str = None,
    size: int = 80,
    **kwargs,
):
    style_order = (
        df_component_analysis[shape_by].unique().tolist()
        if shape_by is not None
        else None
    )
    df_scatter = df_component_analysis.copy()
    df_scatter = df_scatter.groupby(color_by)
    legend_shown = False
    components = df_components.columns.values
    plt.Figure()
    fig, ax = plt.subplots(
        nrows=len(components),
        ncols=len(components),
    )
    fig.set_facecolor("white")
    fig.set_size_inches(18.5, 10.5)
    for r, comp1 in enumerate(components):
        for c, comp2 in enumerate(components.copy()):
            ax_tmp = ax[r][c]
            if r < c:
                fig.delaxes(ax_tmp)
            if r > c:
                sns.scatterplot(
                    x=comp2,
                    y=comp1,
                    hue=join_iterable(color_by)
                    if (isinstance(color_by, str) or len(color_by) == 1)
                    else df_component_analysis[color_by].apply(tuple, axis=1),
                    palette=color_map,
                    s=size,
                    style=shape_by,
                    style_order=style_order,
                    alpha=0.9,
                    # label=join_iterable(label),
                    edgecolor="DarkSlategray",
                    legend=True if r == 1 and c == 0 else False,
                    ax=ax_tmp,
                    data=df_component_analysis,
                )
                # for name, group_data in df_scatter:
                #     if isinstance(name, tuple):
                #         label=[*name]
                #     elif isinstance(name, (str, int, float)):
                #         label=[str(name)]
                #     sns.scatterplot(
                #         x=comp2,
                #         y=comp1,
                #         size=size,
                #         style=shape_by,
                #         alpha=0.9,
                #         label=join_iterable(label),
                #         edgecolor='DarkSlategray',
                #         color=color_map[name],
                #         ax=ax[r][c],
                #         data=group_data,
                #     )
                if r == 1 and c == 0 and legend_shown == False:
                    ax[r][c].legend(bbox_to_anchor=(1 * len(components), 2.3), ncol=3)
                    legend_shown = True
                if r == len(components) - 1:
                    ax_tmp.set_xlabel(comp2, size=16)
                else:
                    ax_tmp.axes.get_xaxis().set_visible(False)
                if c == 0 and r > 0:
                    ax_tmp.set_ylabel(comp1, size=16, labelpad=20)
                else:
                    ax_tmp.axes.get_yaxis().set_visible(False)
            if r == c:
                plot_pie_on_ax(
                    ax_tmp,
                    component=comp1,
                    df_components=df_components,
                    channels=channels,
                    channel_color_map=channel_color_map,
                )
                ax[r][c].title.set_size(16)
    plt.suptitle(title, size=24, x=0.5, y=0.95)
    make_space_above(ax, topmargin=1.5)
    if save_info:
        save_matplotlib_figure(
            fig,
            path=graph_output_folder,
            figurename=figurename,  # df_component_analysis.attrs['name']+'_allcomponentscatter',
            **kwargs,
        )
    # plt.show()
    return fig


def pca_scree_plot(
    pca_explained_var_ratio,
    title: str,
    figurename: str,
    save_info: bool,
    graph_output_folder: str,
    **kwargs,
):
    pc_col = ["PC{}".format(i + 1) for i in range(pca_explained_var_ratio.shape[0])]
    df_explained_variance_ratio = pd.DataFrame(
        pca_explained_var_ratio,
        columns=["Proportion of Explained Variance"],
        index=pc_col,
    )
    df_explained_variance_ratio.attrs["name"] = figurename + "_explained_var"
    print("Explained Variance Ratio")
    print(df_explained_variance_ratio)
    PC_values = np.arange(pca_explained_var_ratio.shape[0]) + 1
    plt.Figure()
    fig, ax = plt.subplots()
    fig.set_facecolor("white")
    fig.set_size_inches(8.5, 8.5)
    ax.plot(PC_values, pca_explained_var_ratio * 100, "ro-", linewidth=2)
    for x, y in zip(PC_values, pca_explained_var_ratio):
        plt.annotate(
            text=str(np.round(y * 100, 1)) + "%", xy=(x + 0.1, y * 100), ha="left"
        )
    plt.title(label=title)
    plt.ylim(bottom=0)
    plt.xticks(PC_values)
    plt.grid(visible=False, axis="x")
    plt.xlabel("Principal Component")
    plt.ylabel("Percent of Variance Explained (%)")
    if save_info:
        save_matplotlib_figure(
            fig, path=graph_output_folder, figurename=figurename, **kwargs
        )
        save_dataframe_to_csv(df_explained_variance_ratio, graph_output_folder)
    # plt.show()
