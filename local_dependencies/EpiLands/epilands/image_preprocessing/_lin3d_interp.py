from numba import jit
import numpy as np
from scipy.interpolate import interp1d
from scipy import ndimage
from typing import List, Tuple, Union
import skimage

# from ._get_object_bbox import get_object_bbox


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
        for i, (idxMin, idxMax) in objectBBox:
            objectBBox[i] = (idxMin + margin, idxMax + margin)
    # for debug
    # img = bwImg.copy()
    # img[
    #             objectBBox[0][0] : objectBBox[0][1],
    #             objectBBox[1][0] : objectBBox[1][1],
    #             objectBBox[2][0] : objectBBox[2][1],
    #         ]=1
    # img = np.where(bwImg==1, 10, img)
    # tiff.imshow(img)
    return objectBBox


def ezresizeimg(orgImg, InitialDims: Tuple, FinalDims: Tuple):
    """
    orgImg: original image (numpy array, dims=[x, y, z]),
    x/y/zinit: the inital lengths of each voxel edge in the same physical units (e.g. microns)
    x/y/zfinal: the final (desired) lengths of each voxel edge in the same physical units (e.g. microns)
    """
    dimSteps = list(InitialDims[i] / FinalDims[i] for i in range(len(orgImg.shape)))
    dimSamples = list(
        np.linspace(0, dimSize - 1, np.floor(dimSize * dimStep).astype(int))
        for dimSize, dimStep in zip(orgImg.shape, dimSteps)
    )
    interpImgShape = [len(sample) for sample in dimSamples]
    interpImg = skimage.transform.resize(orgImg, output_shape=interpImgShape)
    return interpImg


def ezlin3dinterp(orgImg, zxyInitialDims: Tuple, zxyFinalDims: Tuple):
    """
    orgImg: original image (numpy array, dims=[x, y, z]),
    x/y/zinit: the inital lengths of each voxel edge in the same physical units (e.g. microns)
    x/y/zfinal: the final (desired) lengths of each voxel edge in the same physical units (e.g. microns)
    """

    zStep = zxyInitialDims[0] / zxyFinalDims[0]
    xStep = (
        zxyInitialDims[1] / zxyFinalDims[1]
    )  # The ratio of the physical len of vox in z vs x or y (voxel edge length)
    yStep = zxyInitialDims[2] / zxyFinalDims[2]

    matZ, matX, matY = orgImg.shape
    xSample = np.linspace(0, matX - 1, np.floor(matX * xStep).astype(int))
    ySample = np.linspace(0, matY - 1, np.floor(matY * yStep).astype(int))
    zSample = np.linspace(0, matZ - 1, np.floor(matZ * zStep).astype(int))
    return lin3dinterp(orgImg, zSample, xSample, ySample)


# numba no-python compilation
@jit(nopython=True)
def lin3dinterp(orgImg, xSample, ySample, zSample):
    """
    orgImg: original image (numpy array, dims=[x, y, z]),
    xSample: numpy 1D array with values ranging from 0 to original x size, with user defined interval,
    ySample: numpy 1D array with values ranging from 0 to original y size, with user defined interval,
    zSample: numpy 1D array with values ranging from 0 to original z size, with user defined interval

    EX:
        # SEGMENTATION: GENERAL RESHAPE OF INPUT 3D IMAGES
        # Reshape the image dimensions of each voxel via linear interpolation
        # x/y/zinit are the inital lengths of each voxel edge in the same physical units (e.g. microns)
        # x/y/zfinal are the final (desired) lengths of each voxel edge in the same physical units (e.g. microns)

        xSizeRatio = xfinal(um)/xinit(um)
        ySizeRatio = yfinal(um)/yinit(um)
        zSizeRatio = zfinal(um)/zinit(um)

        xStep = 1/xSizeRatio = xinit(um)/xfinal(um)     # The ratio of the physical len of vox in z vs x or y (voxel edge length)
        yStep = 1/ySizeRatio = yinit(um)/yfinal(um)
        zStep = 1/zSizeRatio = zinit(um)/zfinal(um)

        matZ,matX,matY=orgImg.shape
        xSample = np.linspace(0,matX-1,np.floor(matX*xStep).astype(int))
        ySample = np.linspace(0,matY-1,np.floor(matY*yStep).astype(int))
        zSample = np.linspace(0,matZ-1,np.floor(matZ*zStep).astype(int))


        # SEGMENTATION: PRACTICAL RESHAPE OF INPUT 3D IMAGES FOR STARDIST SEGMENTATION
        # Reshape the z-axis without changing X or Y dimensions
        # This can be used to make X Y and Z have 1:1:2 voxel edge length
        zStep = zinit(um)/zfinal(um)    # The ratio of the initial and final physical len of vox in z (voxel edge length in microns)
                                        # In this case, z = 1um and x and y = 0.6um
                                        # so zinit = 1 and zfinal = 1.2 (2 x 0.6)
        zStep = 1/1.2

        matZ,matX,matY=orgImg.shape
        xSample = np.linspace(0,matX-1,matX)
        ySample = np.linspace(0,matY-1,matY)
        zSample = np.linspace(0,matZ-1,np.floor(matZ*zStep).astype(int))



        # FEATURE EXTRACTION: PRACTICAL ISOVOXELIZATION
        # Reshape the z-axis without changing X or Y dimensions
        # This can be used to make X Y and Z have 1:1:1 (equal) voxel edge length
        zStep = zinit(um)/zfinal(um)    # The ratio of the initial and final physical len of vox in z (voxel edge length in microns)
                                        # In this case, z = 1um and x and y = 0.6um
                                        # so zinit = 1 and zfinal = 0.6
        zStep = 1/0.6

        matZ,matX,matY=orgImg.shape
        xSample = np.linspace(0,matX-1,matX)
        ySample = np.linspace(0,matY-1,matY)
        zSample = np.linspace(0,matZ-1,np.floor(matZ*zStep).astype(int))

    """
    interpImg = np.zeros((len(xSample), len(ySample), len(zSample)))
    for i in range(0, len(xSample)):
        for j in range(0, len(ySample)):
            for k in range(0, len(zSample)):
                # get the 8 nearest neighbors
                x1 = int(np.floor(xSample[i]))
                x2 = int(np.ceil(xSample[i]))
                y1 = int(np.floor(ySample[j]))
                y2 = int(np.ceil(ySample[j]))
                z1 = int(np.floor(zSample[k]))
                z2 = int(np.ceil(zSample[k]))
                # if x1==x2 or y1==y2 or z1==z2 then use 1 as the weight
                if x1 == x2:
                    xWeight = 0.5
                else:
                    xWeight = (xSample[i] - x1) / (x2 - x1)
                if y1 == y2:
                    yWeight = 0.5
                else:
                    yWeight = (ySample[j] - y1) / (y2 - y1)
                if z1 == z2:
                    zWeight = 0.5
                else:
                    zWeight = (zSample[k] - z1) / (z2 - z1)

                # interpolate
                interpImg[i, j, k] = (
                    orgImg[x1, y1, z1] * (1 - xWeight) * (1 - yWeight) * (1 - zWeight)
                    + orgImg[x2, y1, z1] * xWeight * (1 - yWeight) * (1 - zWeight)
                    + orgImg[x1, y2, z1] * (1 - xWeight) * yWeight * (1 - zWeight)
                    + orgImg[x1, y1, z2] * (1 - xWeight) * (1 - yWeight) * zWeight
                    + orgImg[x2, y1, z2] * xWeight * (1 - yWeight) * zWeight
                    + orgImg[x1, y2, z2] * (1 - xWeight) * yWeight * zWeight
                    + orgImg[x2, y2, z1] * xWeight * yWeight * (1 - zWeight)
                    + orgImg[x2, y2, z2] * xWeight * yWeight * zWeight
                )

    return interpImg


def ezshapelin3dinterp(labelImg, zxyInitialDims: Tuple, zxyFinalDims: Tuple):
    # The ratio of the physical len of vox in z vs x or y (voxel edge length)
    dimSteps = (
        zxyInitialDims[0] / zxyFinalDims[0],
        zxyInitialDims[1] / zxyFinalDims[1],
        zxyInitialDims[2] / zxyFinalDims[2],
    )
    imgDims = labelImg.shape
    dimSamples = [
        np.linspace(0, dim - 1, np.floor(dim * step).astype(int))
        for dim, step in zip(imgDims, dimSteps)
    ]
    # xSample = np.linspace(0,matX-1,np.floor(matX*xStep).astype(int))
    # ySample = np.linspace(0,matY-1,np.floor(matY*yStep).astype(int))
    # zSample = np.linspace(0,matZ-1,np.floor(matZ*zStep).astype(int))
    interpImg = np.zeros([len(sample) for sample in dimSamples])
    objects = np.unique(labelImg)
    objects = np.delete(objects, np.where(objects == 0))
    for cellIdx in objects:
        # get the binary image of the "celIdxl"-th cell
        objCellLabel = np.where(
            labelImg == cellIdx, 1, 0
        )  # set teh value one to the "celIdxl"-th cell, zero for the others
        objectBBox = get_object_bbox(objCellLabel)
        objectInterpLoc = []
        objectBBoxSamples = []
        objectPaddedBBox = []
        for dim, (idxMin, idxMax) in enumerate(objectBBox):
            dimSampleSpace = dimSamples[dim]
            # find where the sampled linspace is within 1 pixel of the bbox
            # this let's us not calculate all of the distances, ones to the left
            leftOfIdxMin = dimSampleSpace[np.where(dimSampleSpace <= idxMin)]
            # get closest idx (the last idx), ie the length of the array
            nearestToIdxMin = (
                len(leftOfIdxMin) - 1
            )  ####### added -1 because len starts at 1 and we want idx vals
            # get the value at that idx, ie the last value
            # this is the distance of that pixel in original pixel units
            minDistanceFromOrigin = leftOfIdxMin[-1]
            # get the distance to the edge of the bbox
            minDistanceFromBBox = abs(idxMin - minDistanceFromOrigin)
            # ceil gives us the integer value of the distance from bbox
            # we have to pad the cropped image by this margin on this side
            minMargin = np.ceil(minDistanceFromBBox)
            # find the padded index for the left
            paddedIdxMin = idxMin - minMargin
            # repeat for right side
            rightOfIdxMax = dimSampleSpace[np.where(dimSampleSpace >= idxMax - 1)]
            # need to subtract the length from the total, since index is left->right
            nearestToIdxMax = len(dimSampleSpace) - len(
                rightOfIdxMax
            )  #########May or may not need to add 1
            # distance from the left edge of the image is given by first element now
            maxDistanceFromOrigin = rightOfIdxMax[0]
            maxDistanceFromBBox = abs(idxMax - maxDistanceFromOrigin)
            maxMargin = np.ceil(maxDistanceFromBBox)
            # we add the margin to get the padded idx since we are on the right now
            paddedIdxMax = idxMax + maxMargin

            # get the edge len of the cropped image in its own dimensions
            # croppedEdgeLen = paddedIdxMax - paddedIdxMin

            # get the edge len of the interpolated image in its own dimensions
            interpEdgeLen = nearestToIdxMax - nearestToIdxMin  # + 1  # ??
            # get the sample space in cropped-image units for the interpolated image
            croppedSample = np.linspace(
                start=minDistanceFromOrigin - paddedIdxMin,
                stop=maxDistanceFromOrigin - paddedIdxMin,
                num=interpEdgeLen,
            )
            # save the values
            objectBBoxSamples.append(croppedSample)
            objectPaddedBBox.append((int(paddedIdxMin), int(paddedIdxMax)))
            objectInterpLoc.append((nearestToIdxMin, nearestToIdxMax))

        objCellLabel = objCellLabel[
            objectPaddedBBox[0][0] : objectPaddedBBox[0][1],
            objectPaddedBBox[1][0] : objectPaddedBBox[1][1],
            objectPaddedBBox[2][0] : objectPaddedBBox[2][1],
        ]

        interpImgSlice = interpImg[
            objectInterpLoc[0][0] : objectInterpLoc[0][1],
            objectInterpLoc[1][0] : objectInterpLoc[1][1],
            objectInterpLoc[2][0] : objectInterpLoc[2][1],
        ]
        interpMask = shapelin3dinterp(
            mask=objCellLabel,
            zSample=objectBBoxSamples[0],
            xSample=objectBBoxSamples[1],
            ySample=objectBBoxSamples[2],
        )
        labeledImgSlice = np.where(interpMask > 0, cellIdx, interpImgSlice)
        interpImg[
            objectInterpLoc[0][0] : objectInterpLoc[0][1],
            objectInterpLoc[1][0] : objectInterpLoc[1][1],
            objectInterpLoc[2][0] : objectInterpLoc[2][1],
        ] = labeledImgSlice

    return interpImg


def ezresizemask(labelImg, InitialDims: Tuple, FinalDims: Tuple):
    # The ratio of the physical len of vox in z vs x or y (voxel edge length)
    dimSteps = list(InitialDims[i] / FinalDims[i] for i in range(len(labelImg.shape)))
    dimSamples = list(
        np.linspace(0, dimSize - 1, np.floor(dimSize * dimStep).astype(int))
        for dimSize, dimStep in zip(labelImg.shape, dimSteps)
    )
    interpImgShape = list(len(sample) for sample in dimSamples)
    interpImg = np.zeros(interpImgShape)
    objects = np.unique(labelImg)
    objects = np.delete(objects, np.where(objects == 0))
    for cellIdx in objects:
        # get the binary image of the "celIdxl"-th cell
        objCellLabel = np.where(
            labelImg == cellIdx, 1, 0
        )  # set teh value one to the "celIdxl"-th cell, zero for the others
        objectBBox = get_object_bbox(objCellLabel)
        objCellLabel = objCellLabel[
            tuple(slice(dimBBox[0], dimBBox[1]) for _, dimBBox in enumerate(objectBBox))
        ]
        objectInterpLoc = []
        for dim, (idxMin, idxMax) in enumerate(objectBBox):
            dimSampleSpace = dimSamples[dim]
            # find where the sampled linspace is within 1 pixel of the bbox
            # this let's us not calculate all of the distances, ones to the left
            leftOfIdxMin = dimSampleSpace[np.where(dimSampleSpace <= idxMin)]
            # get closest idx (the last idx), ie the length of the array
            nearestToIdxMin = (
                len(leftOfIdxMin) - 1
            )  ####### added -1 because len starts at 1 and we want idx vals
            rightOfIdxMax = dimSampleSpace[np.where(dimSampleSpace >= idxMax - 1)]
            # need to subtract the length from the total, since index is left->right
            nearestToIdxMax = len(dimSampleSpace) - len(
                rightOfIdxMax
            )  #########May or may not need to add 1
            objectInterpLoc.append((nearestToIdxMin, nearestToIdxMax))

        # matZ, matX, matY = objCellLabel.shape
        # maskEdge = objCellLabel.copy().astype(np.float64)
        # # get the edge of the mask using morphological erosion for each slice
        # # and covert it to distance map
        # for z in range(matZ):
        #     # zeropadding the image
        #     tmpMask = np.zeros((matX + 2, matY + 2), dtype=np.uint8)
        #     tmpMask[1:-1, 1:-1] = objCellLabel[z, :, :]
        #     if np.sum(objCellLabel[z, :, :]) == 0:
        #         maskEdge[z, :, :] = -1
        #         continue

        #     maskEdge[z, :, :] = (
        #         ndimage.distance_transform_edt(tmpMask)
        #         - ndimage.distance_transform_edt(1 - tmpMask)
        #     )[1:-1, 1:-1]

        interpImgSlice = interpImg[
            tuple(
                slice(dimLoc[0], dimLoc[1]) for _, dimLoc in enumerate(objectInterpLoc)
            )
        ]
        interpMask = skimage.transform.resize(
            objCellLabel > 0, output_shape=interpImgSlice.shape
        )
        # interpMask = skimage.transform.rescale(maskEdge, scale=dimSteps) #initially used this, but size was diff. Now using resize
        labeledImgSlice = np.where(interpMask > 0, cellIdx, interpImgSlice)
        interpImg[
            tuple(
                slice(dimLoc[0], dimLoc[1]) for _, dimLoc in enumerate(objectInterpLoc)
            )
        ] = labeledImgSlice
    return interpImg


def shapelin3dinterp(mask, zSample, xSample, ySample):
    """
    np.linspace(delta1, delta1+n, n)

    """
    matZ, matX, matY = mask.shape
    maskEdge = mask.copy().astype(np.float64)
    # get the edge of the mask using morphological erosion for each slice
    # and covert it to distance map
    for z in range(matZ):
        # zeropadding the image
        tmpMask = np.zeros((matX + 2, matY + 2), dtype=np.uint8)
        tmpMask[1:-1, 1:-1] = mask[z, :, :]
        if np.sum(mask[z, :, :]) == 0:
            maskEdge[z, :, :] = -1
            continue

        maskEdge[z, :, :] = (
            ndimage.distance_transform_edt(tmpMask)
            - ndimage.distance_transform_edt(1 - tmpMask)
        )[1:-1, 1:-1]

    # get the edge of the mask using morphological erosion for each slice
    mask = lin3dinterp(maskEdge, zSample, xSample, ySample)
    mask = (mask > 0).astype(int)

    return mask


if __name__ == "__main__":
    # import tensorflow.config as tfc

    # gpuIdx = 1
    # gpus = tfc.list_physical_devices(device_type="GPU")
    # tfc.experimental.set_memory_growth(gpus[gpuIdx], True)
    # tfc.set_visible_devices(gpus[gpuIdx], "GPU")

    from stardist.models.model2d import StarDist2D
    from stardist.models.model3d import StarDist3D
    import tifffile as tiff
    import skimage
    from typing import Tuple, Any
    import numpy as np
    import pandas as pd
    import os

    stardist_model_2d = StarDist2D.from_pretrained("2D_versatile_fluo")
    stardist_model_3d = StarDist3D.from_pretrained("3D_demo")

    from skimage.data import cells3d

    img = cells3d()[:, 1, :, :]

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

    def segment_image_stardist3d(image) -> Tuple[np.ndarray, list, dict]:
        masks, details = stardist_model_3d.predict_instances(image)
        # count the unique masks and return objects and mask sizes
        objects, counts = np.unique(masks.reshape(-1, 1), return_counts=True, axis=0)
        # delete 0 as that labels background
        objects = list(np.delete(objects, 0))
        return masks, objects, details

    def segment_image_stardist2d(image) -> Tuple[np.ndarray, list, dict]:
        masks, details = stardist_model_2d.predict_instances(image)
        # count the unique masks and return objects and mask sizes
        objects, counts = np.unique(masks.reshape(-1, 1), return_counts=True, axis=0)
        # delete 0 as that labels background
        objects = list(np.delete(objects, 0))
        return masks, objects, details

    def read_zstack(image_files):
        if not isinstance(image_files, pd.Series):
            image_files = pd.Series(image_files)
        image_files = image_files.sort_values()
        full_stack = [tiff.imread(file) for file in image_files]
        tmpImg = np.empty(shape=(len(image_files), *full_stack[0].shape))
        for i, img in enumerate(full_stack):
            tmpImg[i] = img
        return tmpImg

    # imgdir = "/Volumes/1TBAlexey2/test_data/1_phenix_image"
    imgdir = "/Volumes/1TBAlexey2/test_data/1_IC200_image"
    image_files = [imgdir + "/" + file for file in os.listdir(imgdir)]
    # img = read_zstack(image_files)
    img = tiff.imread(image_files)

    interpImg = ezresizeimg(
        img,
        InitialDims=(0.3, 0.3),
        FinalDims=(1, 1),
    )

    # interpImg = ezlin3dinterp(
    #     img,
    #     zxyInitialDims=(1, 0.6, 0.6),
    #     zxyFinalDims=(1, 0.5, 0.5)
    #     # img,
    #     # zxyInitialDims=(0.290, 0.26, 0.26),
    #     # zxyFinalDims=(1, 0.5, 0.5),
    # )

    ninterpImg = glaylconvert(
        interpImg,
        orgHigh=np.quantile(interpImg, 0.99),
        orgLow=np.quantile(interpImg, 0.01),
        qHigh=1,
        qLow=0,
    )

    # xslice = 100
    # zslice = 30
    # tiff.imshow(img[zslice, :, :])
    # tiff.imshow(img[:, xslice, :])
    xslice1 = 50
    zslice1 = 7
    # xslice1 = 263
    # zslice1 = 7
    # tiff.imshow(ninterpImg[zslice1, :, :])
    # tiff.imshow(ninterpImg[:, xslice1, :])

    # masks, objects, details = segment_image_stardist3d(ninterpImg)
    # tiff.imwrite("/Volumes/1TBAlexey2/test_data/1_phenix_image_masks.tiff", data=masks)
    # masks = tiff.imread("/Volumes/1TBAlexey2/test_data/1_phenix_image_masks.tiff")
    masks, objects, details = segment_image_stardist2d(ninterpImg)
    tiff.imwrite("/Volumes/1TBAlexey2/test_data/1_IC200_image_masks.tiff", data=masks)
    # masks = tiff.imread("/Volumes/1TBAlexey2/test_data/1_IC200_image_masks.tiff")

    interpmask = ezresizemask(
        masks,
        InitialDims=(1, 1),
        FinalDims=(0.3, 0.3)
        # masks,
        # InitialDims=(1, 0.6, 0.6),
        # FinalDims=(0.6, 0.6, 0.6)
        # masks,
        # InitialDims=(0.290, 0.26, 0.26),
        # FinalDims=(0.26, 0.26, 0.26)
    )
    # interpmask = ezshapelin3dinterp(
    #     masks,
    #     zxyInitialDims=(2, 1, 1),
    #     zxyFinalDims=(0.6, 0.6, 0.6)
    #     # masks,
    #     # zxyInitialDims=(0.290, 0.26, 0.26),
    #     # zxyFinalDims=(0.26, 0.26, 0.26),
    #     # masks,
    #     # zxyInitialDims=(1, 0.5, 0.5),
    #     # zxyFinalDims=(0.26, 0.26, 0.26),
    # )
    interpImg2 = ezlin3dinterp(
        interpImg,
        zxyInitialDims=(1, 0.5, 0.5),
        zxyFinalDims=(0.6, 0.6, 0.6)
        # interpImg,
        # zxyInitialDims=(0.290, 0.26, 0.26),
        # zxyFinalDims=(0.26, 0.26, 0.26),
        # interpImg,
        # zxyInitialDims=(1, 0.5, 0.5),
        # zxyFinalDims=(0.26, 0.26, 0.26),
    )

    tiff.imshow(masks)
    tiff.imshow(interpmask)
    # tiff.imshow(interpmask2)
    tiff.imshow(masks[:, xslice1, 260:330])
    tiff.imshow(img[:, xslice1, 260:330])
    tiff.imshow(interpmask[:, xslice1, 260:330])
    tiff.imshow(masks[zslice1, :, :])
    tiff.imshow(interpmask[zslice1, :, :])
    tiff.imshow(masks[:, xslice1, :])
    tiff.imshow(interpmask[:, xslice1, :])
