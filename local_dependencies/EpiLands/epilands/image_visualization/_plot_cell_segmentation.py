from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict
import tifffile as tiff

from ..generic_read_write import save_matplotlib_figure


def plot_cell_segmentation(
    objects: Dict[str, Dict[str, np.ndarray]],
    channel: str,
    path: str,
    filename: str,
    object: str = None,
):
    if object is None:
        objIdxs = pd.Series(objects.keys()).astype(int)
        rng = np.random.Generator()
        object = rng.integers(low=objIdxs.min(), high=objIdxs.max())

    fig, ax = tiff.imshow(objects[object][channel])
    save_matplotlib_figure(fig=fig, path=path, filename=filename)
