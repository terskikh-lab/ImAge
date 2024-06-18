from __future__ import annotations

# Import libraries
import plotly.express as px
import pandas as pd
import os

# relative imports


def save_plotly_figure_files(
    figure,
    path: str,
    figurename: str = "nofigurenamegiven",
    image_formats: list = [".html", ".png"],
    **kwargs,
):
    # other formats
    image_formats = [".html", ".png"]
    for _format in image_formats:
        if _format == ".html":
            # HTML
            name = figurename + ".html"
            figure.write_html(os.path.join(path, name))
        else:
            name = figurename + _format
            figure.write_image(os.path.join(path, name))


def update_trace_centroids(
    df: pd.DataFrame,
    plotly_figure,
    color_by_col: list,
    group_centroids: bool,
    sample_col: str = "Sample",
    sample_marker_size: int = 12,
    centroid_marker_size: int = 24,
    marker_line_color: str = "DarkSlateGrey",
    marker_line_width=1,
    **kwargs,
):
    # for troubleshooting
    # MIEL_timeline.for_each_trace(lambda trace:  print(trace.customdata))#[int((np.where(np.array(trace.customdata) == [trace.name+'_ave']))[0])]))#, np.array(trace.customdata) == [trace.name+'-ave'], trace.name))

    groupby_marker_info = df.groupby(color_by_col)

    # below loop finds the location of the group centroid and gives it a different size
    if group_centroids == True:
        info = {}
        for group in groupby_marker_info.groups:
            ave_index = groupby_marker_info.get_group(group)[sample_col].str.contains(
                "ave"
            )
            ave_index_loc = ave_index.index.get_loc(
                ave_index[ave_index == True].index[0]
            )
            info[group] = (
                [sample_marker_size] * ave_index_loc
                + [centroid_marker_size]
                + [sample_marker_size] * (len(ave_index) - (ave_index_loc + 1))
            )

        plotly_figure.for_each_trace(
            lambda trace: trace.update(
                marker_size=info[trace.name],
                marker_line_color=marker_line_color,
                marker_line_width=marker_line_width,
            )
        )
    else:
        plotly_figure.for_each_trace(
            lambda trace: trace.update(
                marker_size=sample_marker_size,
                marker_line_color=marker_line_color,
                marker_line_width=marker_line_width,
            )
        )


def plotly_scatterplot(
    df: pd.DataFrame,
    graph_output_folder: str,
    *,
    x_label: str = None,
    y_label: str = None,
    zeroline: bool = False,
    color_by_col: str = "Group",
    color_map: dict = None,
    show_centroids: bool = False,
    group_col_list: list = None,
    sample_BioID: list = None,
    show_group_centroids: bool = False,
    save_graph: bool = False,
    image_formats: list = [".html", ".png"],
    **kwargs,
):
    if "title" not in kwargs:
        kwargs["title"] = df.attrs["name"]
    if type(df.index) == pd.MultiIndex:
        df = df.extract_multiindex_values(df, axis="rows")
    if show_centroids:
        if group_col_list == None or sample_BioID == None:
            raise ValueError(
                "plotly_scatterplot:\
                if show_centroids = True you must provide a group_col_list and a sample_BioID"
            )
        df = calculate_sample_centroids_from_bins(
            df,
            group_col_list,
            sample_BioID,
            calculate_group_centroids=show_group_centroids,
        )

    scatter = px.scatter(
        df,
        x=x_label,
        y=y_label,
        color=color_by_col,
        hover_data={i: True for i in df.columns},
        color_discrete_map=color_map,
    )

    if show_centroids:
        update_trace_centroids(
            df.extract_multiindex_values(df, axis="rows"),
            scatter,
            color_by_col,
            show_group_centroids,
            **kwargs,
        )

    scatter.update_traces(
        marker_line_color="DarkSlateGrey",
        marker_line_width=1,
    )
    scatter.update_xaxes(
        showgrid=False, showline=True, linewidth=2, linecolor="black", ticks="outside"
    )
    scatter.update_yaxes(
        showgrid=False, showline=True, linewidth=2, linecolor="black", ticks="outside"
    )

    if zeroline:
        scatter.update_yaxes(zeroline=True, zerolinecolor="black", zerolinewidth=2)

    scatter.update_layout(
        showlegend=True,
        legend_title_text="Experimental Group",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="white",
    )
    scatter.update_yaxes(scaleanchor="x", scaleratio=1)

    scatter.show()
    if save_graph:
        save_plotly_figure_files(
            scatter,
            figurename="{}_scatter".format(df.attrs["name"]),
            image_formats=image_formats,
            path=graph_output_folder,
        )

    return scatter
