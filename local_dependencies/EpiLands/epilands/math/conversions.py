import numpy as np


def pixel_area_to_diameter(x):
    return np.sqrt(4 * x / np.pi)
