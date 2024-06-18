# import libraries
from typing import Union
import logging
import matplotlib.markers
import matplotlib.pyplot as plt
import numpy as np
import re

# from IPython.display import display

# relative imports
from ..config import (
    UNQIUE_MARKERS_,
    DEFAULT_CHANNEL_PALETTE,
    DEFAULT_CATEGORITCAL_PALETTE,
    DEFAULT_CATEGORICAL_CONTINUOUS_PALETTE,
    DEFAULT_CONTINUOUS_COLOR_PALETTE,
    DEFAULT_SHAPE_PALETTE,
)
from ..tools import join_iterable

logger = logging.getLogger(__name__)

import logging

import pandas as pd
import numpy as np


def map_keys_to_items(keys, items):
    # create an empty dictionary to store the mapping
    mapping = {}
    # if the number of keys is greater than the number of items, log a warning message
    if len(keys) > len(items):
        logging.warning(
            "Number of keys is greater than number of items. There will be redundant items in the mapping."
        )
    # iterate over the keys
    for i, key in enumerate(keys):
        # map the key to the item at the corresponding index in the items list
        mapping[key] = items[i % len(items)]
    return mapping


def create_new_mapping_match_keys(groups: list, match_map: dict):
    """Creates a new mapping by checking if keys in match_map are contained in the groups"""
    new_color_map = {}
    for group in groups:
        for key in match_map:
            if key in join_iterable(group):
                if group in new_color_map.keys():
                    logger.warning(
                        f"{group} matches multiple keys in mapping {match_map}"
                    )
                    logger.warning(f"updating mapping to: {group}={match_map[key]}")
                new_color_map[group] = match_map[key]
    return new_color_map


def create_channel_mapping(
    channels: list,
    package: str = "plt",
    channel_color_pallete=DEFAULT_CHANNEL_PALETTE,
) -> dict:
    """
    Creates a color pallete for the given channels.

    channels: list of strings

    package: str chosen package for color pallete. Can be one of 'px', 'plt', or 'both'

    channel_color_pallete: list of numpy arrays containing color values
    """
    if package == "plt":
        channel_color_map_plt = map_keys_to_items(
            keys=channels, items=channel_color_pallete
        )
        # display(channel_color_map_plt)
        return channel_color_map_plt
    if package == "px":
        channel_color_pallete = transform_cmap_plotly_to_matplot(
            channel_color_pallete, output_format="px"
        )
        channel_color_map_px = map_keys_to_items(
            keys=channels, items=channel_color_pallete
        )
        # display(channel_color_map_px)
        return channel_color_map_px
    else:
        raise NotImplementedError(
            "Only packages matplotlib (plt) and plotly.express (px) currently suported"
        )


def create_color_mapping(
    legend_items: list,
    package: str,
    map_type: str,
) -> dict:
    """
    creates a color mapping for the legend items given.

    legend_items: list of strings or tuples (tuples will automatically be converted to strings)

    package: str chosen package for color pallete. Can be one of 'px', 'plt', or 'both'

    map_type: str whether or not to use a continuous or categorical color palette
    """
    plt_to_px = lambda t: "rgb(" + ",".join([str(int(i * 255)) for i in t]) + ")"
    if package not in ["px", "plt"]:
        raise NotImplementedError(
            "create_marker_mapping: package must be one of "
            + "'px' or 'plt' but {} was given".format(package)
        )
    if map_type == "categorical":
        if len(legend_items) > len(DEFAULT_CATEGORITCAL_PALETTE.colors):
            print("WARNING: CHANGING CMAP BECAUSE TOO MANY GROUPS WERE DETECTED")
            color_pallette = DEFAULT_CATEGORICAL_CONTINUOUS_PALETTE(
                np.linspace(0, 1, len(legend_items))
            )
        else:
            color_pallette = DEFAULT_CATEGORITCAL_PALETTE(
                np.linspace(0, 1, len(legend_items))
            )
    if map_type == "continuous":
        color_pallette = DEFAULT_CONTINUOUS_COLOR_PALETTE(
            np.linspace(0, 1, len(legend_items))
        )
    if package == "px":
        color_pallette = list(map(plt_to_px, color_pallette))
    color_map = map_keys_to_items(keys=legend_items, items=color_pallette)
    return color_map


def create_shape_mapping(legend_items: list, shapes: list = DEFAULT_SHAPE_PALETTE):
    if len(legend_items) > len(shapes):
        logger.warning(
            "Too many legend items given for shape. Using full marker styles from matplotlib"
        )
        shapes = list(matplotlib.markers.MarkerStyle.markers.keys())
        if len(legend_items) > len(shapes):
            logger.warning(
                "Still too many legend items given for shape.",
                "\nmatplotlib markers cannot account for all groups.",
                "\nlenfth of legend_items must be less than {} for no redundancy".format(
                    len(shapes)
                ),
                "\nExpect redundant marker mappings",
            )
    shape_map = map_keys_to_items(keys=legend_items, items=shapes)
    # display(shape_map)
    return shape_map


def label_by_metadata(
    df: pd.DataFrame, label: str, metadata_cols: list, mapping: dict = None
):
    # create an empty list to store the color labels
    labels = []
    # if a color map is not provided, create an empty dictionary to store the mapping of metadata tuples to color labels
    if mapping is None:
        mapping = {}
    # iterate over the rows of the data frame
    for index, row in df.iterrows():
        # create an empty list to store the metadata values for this row
        metadata_values = []
        # iterate over the metadata columns
        for col in metadata_cols:
            # append the value of this metadata column for this row to the list of metadata values
            metadata_values.append(row[col])
        # convert the list of metadata values to a tuple
        metadata_tuple = tuple(metadata_values)
        # if a color mapping for this metadata tuple is provided, use it
        if metadata_tuple in mapping:
            labels.append(mapping[metadata_tuple])
        # if a color mapping is not provided, generate a new color label
        else:
            # generate a new color label
            color_label = np.random.rand(
                3,
            )
            # add the metadata tuple and color label to the color map
            mapping[metadata_tuple] = color_label
            # append the color label to the list of color labels
            labels.append(color_label)
    # add the list of color labels to the data frame as a new column
    df[label] = labels
    return df


def transform_cmap_plotly_to_matplot(color_pallete, output_format: str):
    if output_format not in ["plt", "px"]:
        raise NotImplementedError(
            "transform_cmap_plotly_to_matplot: output_format must be one of {} but {} was given".format(
                ["plt", "px"], output_format
            )
        )
    px_to_plt = lambda s: tuple(
        int(i) / 255 for i in re.findall(re.compile("[0-9]+"), s)
    )
    plt_to_px = lambda t: "rgb(" + ",".join([str(int(i * 255)) for i in t]) + ")"
    if output_format == "plt":
        return [px_to_plt(color) for color in color_pallete]
    if output_format == "px":
        return [plt_to_px(color) for color in color_pallete]


def make_space_above(axes: plt.Axes, topmargin: float = 1):
    """increase figure size to make topmargin (in inches) space for
    titles, without changing the axes sizes
    from https://stackoverflow.com/questions/55767312/how-to-position-suptitle"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1 - s.top) * h + topmargin
    fig.subplots_adjust(bottom=s.bottom * h / figh, top=1 - topmargin / figh)
    fig.set_figheight(figh)
