from __future__ import annotations
from matplotlib import axes

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from stardist.plot import render_label
from typing import Tuple

from ..generic_read_write import save_matplotlib_figure


def show_thresholded_mask(
    image: np.ndarray,
    masks: np.ndarray,
    masks_in: np.ndarray,
    masks_out: np.ndarray,
    output_directory: str,
    save: bool,
) -> Tuple[Figure, Tuple[Axes, Axes, Axes]]:
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(8, 6),
        # dpi=DEFAULT_DPI
    )
    masked_img = render_label(masks, img=image, cmap=(150, 150, 0))
    axs[0].imshow(masked_img)
    axs[0].axis("off")
    axs[0].set_title("No Thresholding")
    masked_img2 = render_label(masks_in, img=image, cmap=(0, 255, 0))
    axs[1].imshow(masked_img2)
    axs[1].axis("off")
    axs[1].title("Post Thresholding")
    masked_img3 = render_label(masks_out, img=image, cmap=(255, 0, 0))
    axs[2].imshow(masked_img3)
    axs[2].axis("off")
    axs[2].title("Thresholded Objects")
    if save == True:
        save_matplotlib_figure(
            fig, path=output_directory, filename="thresholded_objects_mask_overlay"
        )
    return fig, axs
