import numpy as np
from typing import List, Tuple


def get_object_bbox(bwImg: np.ndarray, margin: int = None) -> List[Tuple]:
    # get the size of the image
    dims = bwImg.shape
    # initialize a list of tuples for storing idxMin and idxMax values
    objectBBox = []
    for i, dim in enumerate(dims):
        # get a maximum and minimum column coordinate
        coordinate = np.arange(0, dim)
        idx = np.any(
            a=(bwImg == 1), axis=tuple(j for j, _ in enumerate(dims) if j != i)
        )
        idxMin = coordinate[idx][0]
        idxMax = coordinate[idx][-1] + 1
        objectBBox.append((idxMin, idxMax))
    if margin is not None:
        # add the margin to the bbox
        for i, (idxMin, idxMax) in enumerate(objectBBox):
            objectBBox[i] = (idxMin + margin, idxMax + margin)
    return objectBBox
