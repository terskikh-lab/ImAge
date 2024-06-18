# import libraries
import matplotlib.pyplot as plt
import umap.plot

# RELATIVE IMPORTS #
from ..generic_read_write import save_matplotlib_figure


def plot_umap_scatter(
    umap_mapper,
    labels,
    title: str,
    figurename: str,
    color_map,
    save_info: bool,
    graph_output_folder: str,
    **kwargs
):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_facecolor("white")
    umap.plot.points(umap_mapper, labels=labels, color_key=color_map, ax=ax, **kwargs)
    plt.legend()
    plt.title(title)
    if save_info:
        save_matplotlib_figure(fig, path=graph_output_folder, figurename=figurename)
    # plt.show()
    return fig, ax
