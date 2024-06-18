import numpy as np
from ._get_object_bbox import get_object_bbox

# Description cellselecter
# separate cells from the image
# Kenta Ninomiya @ Kyushu University: 2021/07/29
def cellselecter(img: np.ndarray, label: np.ndarray, margin: int, cellIdx: int):
    # get the binary image of the "celIdxl"-th cell
    objCellLabel = np.where(
        label == cellIdx, 1, 0
    )  # set teh value one to the "celIdxl"-th cell, zero for the others

    # get the size of the image
    [rowNum, colNum] = objCellLabel.shape

    # get a maximum and minimum row coordinate
    coordinateRow = np.arange(0, rowNum)
    idxRow = np.any(a=objCellLabel == 1, axis=int(1))

    # get a maximum and minimum column coordinate
    coordinateCol = np.arange(0, colNum)
    idxCol = np.any(a=objCellLabel == 1, axis=int(0))

    rowMin = coordinateRow[idxRow][0]
    rowMax = coordinateRow[idxRow][-1]
    colMin = coordinateCol[idxCol][0]
    colMax = coordinateCol[idxCol][-1]

    # slicing the matrix
    objImg = img[rowMin : rowMax + 1, colMin : colMax + 1]
    objImg = np.pad(objImg, [margin, margin], "constant")

    objCellLabel = objCellLabel[rowMin : rowMax + 1, colMin : colMax + 1]
    objCellLabel = np.pad(objCellLabel, [margin, margin], "constant")

    return objImg, objCellLabel


# Description cellselecter
# separate cells from the image
# Kenta Ninomiya @ Kyushu University: 2021/07/29
def cellselecter_ND(
    img: np.ndarray, 
    label: np.ndarray, 
    margin: int, 
    cellIdx: int,
    ):
    # get the binary image of the "celIdxl"-th cell
    objCellLabel = np.where(
        label == cellIdx, 1, 0
    )  # set teh value one to the "celIdxl"-th cell, zero for the others
    objectBBox = get_object_bbox(objCellLabel)
    # slicing the matrix
    slices = tuple(slice(i, j) for (i, j) in objectBBox)
    objImg = img[slices]
    objImg = np.pad(objImg, [margin, margin], "constant")
    objCellLabel = objCellLabel[slices]
    objCellLabel = np.pad(objCellLabel, [margin, margin], "constant")
    return objImg, objCellLabel
