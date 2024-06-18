from stardist.models.model2d import StarDist2D
from stardist.models.model3d import StarDist3D
from typing import Tuple, Any
import numpy as np

stardist_model_2d = StarDist2D.from_pretrained("2D_versatile_fluo")
stardist_model_3d = StarDist3D.from_pretrained("3D_demo")



def segment_image_stardist2d(image) -> Tuple[np.ndarray, list, dict]:
    masks, details = stardist_model_2d.predict_instances(image)
    # count the unique masks and return objects and mask sizes
    objects, counts = np.unique(masks.reshape(-1, 1), return_counts=True, axis=0)
    # delete 0 as that labels background
    objects = list(np.delete(objects, 0))
    return masks, objects, details



def segment_image_stardist3d(image) -> Tuple[np.ndarray, list, dict]:
    masks, details = stardist_model_3d.predict_instances(image)
    # count the unique masks and return objects and mask sizes
    objects, counts = np.unique(masks.reshape(-1, 1), return_counts=True, axis=0)
    # delete 0 as that labels background
    objects = list(np.delete(objects, 0))
    return masks, objects, details
