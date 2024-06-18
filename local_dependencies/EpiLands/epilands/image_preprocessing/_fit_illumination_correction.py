import basicpy
import numpy as np


def fit_illumination_correction_model(group_images: np.array):
    correction_model = basicpy.BaSiC()
    correction_model.get_darkfield = True
    correction_model.fit(group_images)
    return correction_model
