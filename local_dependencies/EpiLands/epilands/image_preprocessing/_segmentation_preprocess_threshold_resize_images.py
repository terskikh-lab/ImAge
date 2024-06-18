import numpy as np
from typing import Union

from ._resize_image import resize_image


def segmentation_preprocess_threshold_resize_images(
    image: np.ndarray, threshold: Union[int, float], resize_factor: Union[int, float]
) -> np.ndarray:
    threshImg = np.where(image > threshold, threshold, image)
    threshResizeImg = resize_image(threshImg, resize_factor=resize_factor)
    return threshResizeImg
