import numpy as np
from typing import Union


# Description glaylconvert
# img ==> numpy array
# Kenta Ninomiya @ Kyushu University: 2021/3/19
def glaylconvert(
    img: np.ndarray,
    orgLow: Union[int, float],
    orgHigh: Union[int, float],
    qLow: Union[int, float],
    qHigh: Union[int, float],
) -> np.ndarray:
    """
    glaylconvert: convert gray scale image to q-space
    """
    # Quantization of the grayscale levels in the ROI
    img = np.where(img > orgHigh, orgHigh, img)
    img = np.where(img < orgLow, orgLow, img)
    cImg = ((img - orgLow) / (orgHigh - orgLow)) * (qHigh - qLow) + qLow
    return cImg
