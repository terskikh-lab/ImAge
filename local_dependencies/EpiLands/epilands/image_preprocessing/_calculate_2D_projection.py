import numpy as np
import tifffile as tiff


def calculate_2D_projection(files: list, method: str = "max") -> np.ndarray:
    zStackImg = tiff.imread(files)
    if method == "max":
        zStackProj = zStackImg.max(axis=0)
    else:
        raise NotImplementedError(f"{method} has not been implemented")
    return zStackProj


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    imgdir = "/Volumes/1TBAlexey2/test_data/1_phenix_image"
    image_files = [imgdir + "/" + file for file in os.listdir(imgdir)]

    img = calculate_2D_projection(image_files)
