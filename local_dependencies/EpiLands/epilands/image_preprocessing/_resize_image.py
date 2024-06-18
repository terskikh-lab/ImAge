import numpy as np
from PIL import Image
from typing import Union


def resize_image(image: np.ndarray, resize_factor: Union[int, float]) -> np.ndarray:
    """
    resizes an image by a factor of resize_factor
    """
    return np.array(
        Image.fromarray(image).resize((int(i * resize_factor) for i in image.shape))
    )
