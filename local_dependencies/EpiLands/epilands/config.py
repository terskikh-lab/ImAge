from sre_parse import SPECIAL_CHARS
import os
import numpy as np
import matplotlib.pyplot as plt

NAME_SEPARATOR_ = "_"
COLUMN_SEPARATOR = "_"
DEFAULT_IMAGE_FORMATS = ["pdf"]
DEFAULT_DPI = 400


_fov_consecutive_to_rowcol_dict = {
    "1": "11",
    "2": "12",
    "3": "13",
    "4": "21",
    "5": "22",
    "6": "23",
    "7": "31",
    "8": "32",
    "9": "33",
}

_current_dir = os.path.dirname(os.path.abspath(__file__))

ALL_SUBDIRS = [
    d
    for d in os.listdir(_current_dir)
    if (os.path.isdir(os.path.join(_current_dir, d)) and not d.startswith("__"))
]


COLUMN_SEPARATOR_ = "_"

NAME_SEPARATOR_ = "_"

DEFAULT_IMAGE_FORMATS_ = [".pdf"]

DEFAULT_DPI_ = 400

DISPLAY_PLOTS = False

DEFAULT_CHANNEL_PALETTE = [
    [
        np.array([0.3764705882352941, 0.10196078431372549, 0.2901960784313726]),
        np.array([0.9333333333333333, 0.26666666666666666, 0.1843137254901961]),
        np.array([0.38823529411764707, 0.6745098039215687, 0.7450980392156863]),
        np.array([0.9764705882352941, 0.9568627450980393, 0.9254901960784314]),
    ]
]

DEFAULT_CATEGORITCAL_PALETTE = plt.get_cmap("tab20")
DEFAULT_CATEGORICAL_CONTINUOUS_PALETTE = plt.get_cmap("nipy_spectral")
DEFAULT_CONTINUOUS_COLOR_PALETTE = plt.get_cmap("inferno")

UNQIUE_MARKERS_ = {
    "o": "circle",
    "v": "triangle_down",
    "2": "tri_up",
    "8": "octagon",
    "s": "square",
    "p": "pentagon",
    "*": "star",
    "H": "hexagon2",
    "+": "plus",
    "D": "diamond",
    "d": "thin_diamond",
    "|": "vline",
    "_": "hline",
    "P": "plus_filled",
    "X": "x_filled",
}

DEFAULT_SHAPE_PALETTE = list(UNQIUE_MARKERS_.keys())

# age_color_pallete = {
#     'Young':'#A9D18E',
#     'Old':'#A6A6A6',
#     'YoungDoxo':'#6cf091',
#     'YoungNoDoxo':'#8FA9DC',
#     'OldDoxo':'#afbfde',
#     'OldNoDoxo':'#d3def5',
#     'MiddleAge':'#8FA9DC',
#     '18':'#afbfde',
#     '24':'#d3def5',
#     'NoTumor':'#6cf091',
#     'LiverTumor':'#bb8fdc',
# }

age_color_pallete = {
    "Young": "#3FE336",  # green
    "Old": "#4D4D4D",  # dark gray / steel
    "YoungDoxo": "#6cf091",
    "YoungNoDoxo": "#8FA9DC",
    "OldDoxo": "#afbfde",
    "OldNoDoxo": "#d3def5",
    "MiddleAge": "#ADADAD",  # gray
    "18": "#ADADAD",  # gray
    "24": "#807D7D",  # darker gray
    "NoTumor": "#6cf091",
    "LiverTumor": "#bb8fdc",
}


ab_color_pallete = {
    "DAPI": "#2F5598",
    "H3K27ac": "#FF0A00",
    "H3K27me3": "#00B050",
    "H3K4me1": "#FFC001",
}


OVERWRITE = True
