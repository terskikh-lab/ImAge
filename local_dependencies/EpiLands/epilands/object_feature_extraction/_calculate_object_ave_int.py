import numpy as np
from numba import njit

def calculate_object_ave_int(img: np.ndarray, mask: np.ndarray) -> float:
    try:
        return np.mean(img, where=mask>0)
    except:
        return np.NAN

# @njit
# def calculate_object_ave_int(img: np.ndarray) -> float:
#     try:
#         obj_bounds = np.where(img > 0)
#         obj = img[obj_bounds[0], :]
#         obj = obj[:, obj_bounds[1]]
#         return np.mean(obj)
#     except:
#         return np.NAN


# @njit
# def calculate_object_ave_int(
#     img: np.ndarray,
#     masks: np.ndarray,
#     objectIdx: int) -> float:
#     obj_bounds = np.where(masks==objectIdx)
#     obj = img[obj_bounds[0], obj_bounds[1]]
#     return np.mean(obj)
