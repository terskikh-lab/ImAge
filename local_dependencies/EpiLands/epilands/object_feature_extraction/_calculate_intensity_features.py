import pandas as pd
import numpy as np
from typing import Dict
from ._calculate_object_ave_int import calculate_object_ave_int


def calculate_intensity_features(image_data: Dict[str, np.ndarray]) -> pd.Series:
    object_intensity_features = {
        f"{ch}_object_average_intensity":calculate_object_ave_int(
            img=obj_img, mask=image_data["masks"]
            )
        for ch, obj_img in image_data.items() if ch != "masks"
    }
    return pd.Series(object_intensity_features)
