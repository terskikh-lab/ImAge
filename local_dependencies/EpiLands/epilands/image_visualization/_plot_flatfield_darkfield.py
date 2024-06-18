from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from typing import Tuple, Union

from ..generic_read_write import save_matplotlib_figure


def plot_flatfield_darkfield(
    channel_fov: Union[int, str],
    flatfield: np.ndarray,
    darkfield: np.ndarray,
    output_directory: str,
    filename: str,
    save: bool,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
        # dpi=DEFAULT_DPI
    )
    ax1.imshow(flatfield, cmap="gray")
    ax1.set_title(f"{channel_fov} Flatfield")
    ax2.imshow(darkfield, cmap="gray")
    ax2.set_title(f"{channel_fov} Darkfield")
    if save == True:
        save_matplotlib_figure(fig, path=output_directory, filename=filename)
    return fig, (ax1, ax2)
