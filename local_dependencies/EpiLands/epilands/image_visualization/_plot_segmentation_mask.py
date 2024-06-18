from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from stardist.plot import render_label
from typing import Tuple

from ..generic_read_write import save_matplotlib_figure


def plot_segmentation_mask(
    image: np.array,
    masks: np.array,
    output_directory: str,
    name: str = "segmented_image",
    save: bool = True,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(4, 8),
        # dpi=DEFAULT_DPI
    )
    ax1.imshow(image, cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("input image")
    masked_img = render_label(masks, img=image)
    ax2.imshow(masked_img, cmap="gray")
    ax2.set_axis_off()
    ax2.set_title("Segmentation + input overlay")
    if save == True:
        save_matplotlib_figure(fig, path=output_directory, filename=name)
    return fig, (ax1, ax2)
