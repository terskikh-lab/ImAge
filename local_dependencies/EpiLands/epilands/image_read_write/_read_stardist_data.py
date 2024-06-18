import h5py
import numpy as np
from typing import Tuple


def read_stardist_data(segmentationFile) -> Tuple[dict, dict, list]:
    with h5py.File(segmentationFile, "r") as segmentationFile:
        details = {}
        for detail in segmentationFile["details"].keys():
            details[detail] = np.array(segmentationFile["details"][detail][:])
        # Construct objects linspace since size thresholding has not been done yet
        objects = [*map(int, segmentationFile["image_data"]["masks"].keys())]
        masks = np.array(segmentationFile["fullmask"]["masks"][:])
    return masks, details, objects
