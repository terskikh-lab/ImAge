from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt
from typing import Optional

# Relative imports


def multiotsu(data, *args, **kwargs):
    thresholds = threshold_multiotsu(data, **kwargs)
    return thresholds


    # if DISPLAY_PLOTS:hist_kwargs={}, 
    #     plt.figure()
    #     plt.hist(x=data, bins=100, **hist_kwargs)
    #     plt.vlines(
    #         x=thresholds,
    #         ymin=[0] * len(thresholds),
    #         ymax=[100] * len(thresholds),
    #         colors="r",
    #     )
    #     plt.show()
