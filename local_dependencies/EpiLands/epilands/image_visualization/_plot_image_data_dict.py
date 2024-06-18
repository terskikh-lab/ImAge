from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from typing import Tuple, Union

from ..generic_read_write import save_matplotlib_figure
from ..image_preprocessing import glaylconvert


def plot_image_data_dict(
    image_dict: dict,
    name: Union[str, int],
    output_directory: str,
    save: bool,
    cmap: str = "viridis",
) -> Tuple[Figure, Axes]:
    # display the image data
    fig, axs = plt.subplots(
        1,
        len(image_dict.keys()),
        figsize=(5, 5 * len(image_dict.keys())),
        # dpi=DEFAULT_DPI
    )
    for i, key in enumerate(image_dict.keys()):
        axs[i].imshow(
            glaylconvert(
                img=image_dict[key],
                orgLow=np.percentile(image_dict[key], 1),
                orgHigh=np.percentile(image_dict[key], 99),
                qLow=0,
                qHigh=1,
            ),
            cmap=cmap,
        )
        axs[i].set_axis_off()
        axs[i].set_title(f"{key}")
    if save == True:
        save_matplotlib_figure(fig, path=output_directory, filename=name)
    return fig, axs
