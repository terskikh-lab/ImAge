import numpy as np
import logging
from scipy.signal import convolve
import os
import tensorflow as tf


sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

# ======================================
# Feature Extraction Functions
# ======================================
# Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
# ADAPTED FROM: 1) Mahotas package v: mahotas\features\tas.py
#               2) https://github.com/DWALab/Schormann-et-al/blob/master/MBF_texture_suite_b2.proc.txt
# ======================================
# GLOBAL VARS:

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the TAS feature extraction from the Mahotas package.
# See notes below for why this was edited.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # for 2D
# # _M2 = a 3x3 matrix with all 1s and 10 in the middle
# # _bins2 = np.array([0,1,2,3,4,5,6,7,8,9,10])
# _M2 = np.ones((3, 3))
# _M2[1, 1] = 10
# _bins2 = np.arange(11)

# # for 3D
# # _M3 = 3 3x3 matrices with all 1s and 28 in the middle
# # _bins3 = np.array([0,1,2,3,4,5,6,7,8,9,10])
# _M3 = np.ones((3, 3, 3))
# _M3[1,1,1] = _M3.sum() + 1
# _bins3 = np.arange(28)
# # if IMG == 2D
# if len(img.shape) == 2:
#     M = _M2
#     bins = _bins2
#     saved = 9
# # if IMG == 3D
# elif len(img.shape) == 3:
#     M = _M3
#     bins = _bins3
#     saved = 27
# else:
#     raise ValueError('mahotas.tas: Cannot compute TAS for image of %s dimensions' % len(img.shape))

# def _ctas(img) -> np.ndarray:
#     # Convolve the image with the TAS Kernel M
#     V = convolve(img.astype(np.uint8), M)
#     # Count the values for each bin (0-9, ie number of neighbors)
#     values,_ = np.histogram(V, bins=bins)
#     # Slice out only the values up to 9
#     values = values[:saved] ########## using this would count the number of pixels with value=0 whose
#                             ########## neighbors have value=1, I beleive this is a mistake. See original TAS paper
#                             ########## see _ctas_mod below for a new implementation
#                             ########## - Martin Alvarez-Kuglen, Sanford Burnham Prebys Medical Discovery Institute
#     # Sum the values
#     s = values.sum()
#     if s > 0:
#         # return the normalized number
#         return values/float(s)
#     # else return 0s
#     return values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

USE_TF = False
if USE_TF == False:
    # Create 3X3 kernel for 2D image
    _TAS_2Dkernel = np.ones((3, 3))
    # Set center of the kernel to +1 greater than the sum of the kernel: 9+1=10.
    # NOTE: This allows us to easily count only pixels with
    # NOTE: initial value of 1 (1*10=10) vs pixels with initial value of 0 (0*10=0)
    # NOTE: See _TAS_binedges2D for how we do this
    _TAS_2Dkernel[1, 1] = _TAS_2Dkernel.sum() + 1
    # Modified Bin edges for similarity to original TAS paper
    # Note that the bin edges are of unit length.
    # This allows for density=True to give the proper output.
    # 0 neighbors = 10, 1 neighbor = 11..., 8 neighbors = 18
    # NOTE: np.arange exludes the right edge (last value = 18.5)
    # NOTE: so even though we use 19.5 we will get the right output
    _TAS_binedges2D = np.arange(9.5, 19.5)
    # Create 3X3X3 kernel for 3D image
    _TAS_3Dkernel = np.ones((3, 3, 3))
    # Set center of the kernel to +1 greater than the sum of the kernel: 27+1=28.
    # NOTE: This allows us to easily count only pixels with
    # NOTE: initial value of 1 (1*28=28) vs pixels with initial value of 0 (0*28=0)
    # NOTE: See _TAS_binedges3D for how we do this
    _TAS_3Dkernel[1, 1, 1] = _TAS_3Dkernel.sum() + 1
    # Modified Bin edges for similarity to original TAS paper
    # Note that the bin edges are of unit length.
    # This allows for density=True to give the proper output.
    # 0 neighbors = 28, 1 neighbor = 29..., 26 neighbors = 54
    # NOTE: np.arange exludes the right edge (last value = 54.5)
    # NOTE: so even though we use 55.5 we will get the right output
    _TAS_binedges3D = np.arange(27.5, 55.5)

if USE_TF == True:
    _TAS_2Dkernel = np.ones((3, 3, 1, 1))
    _TAS_2Dkernel[
        1,
        1,
        0,
        0,
    ] = (
        _TAS_2Dkernel.sum() + 1
    )
    _TAS_binedges2D = np.arange(9.5, 19.5)
    _TAS_3Dkernel = np.ones((3, 3, 3, 1, 1))
    _TAS_3Dkernel[1, 1, 1, 0, 0] = _TAS_3Dkernel.sum() + 1
    _TAS_binedges3D = np.arange(27.5, 55.5)

    _TAS_2Dkernel = tf.constant(_TAS_2Dkernel.astype(np.int32))
    _TAS_3Dkernel = tf.constant(_TAS_3Dkernel.astype(np.int32))


def _ctas_mod(
    bwimg: np.ndarray | tf.Tensor, kernel: np.ndarray, counting_bin_edges: np.ndarray
) -> np.ndarray:
    # Convolve the image with the TAS Kernel
    #    V=None
    if USE_TF == False:
        V = convolve(bwimg.astype(np.uint8), kernel, mode="same")
    elif USE_TF == True:
        pad = np.array([[0, 0]] * len(bwimg.shape))
        for i, d in enumerate(bwimg.shape):
            j = 0
            while (d + j) % 3 != 0:
                j += 1
            if j > 0:
                pad[i][0] = j
        # if any(0 in p for p in pad):
        bwimg = np.pad(bwimg, pad)
        bwimg = tf.constant(np.expand_dims(bwimg.astype(np.int32), axis=(0, -1)))
        bwimg = tf.cast(bwimg, tf.int32)
        V = tf.nn.convolution(
            input=bwimg, filters=kernel, padding="SAME", data_format="NHWC"
        ).numpy()

    # Count the values for each bin (10-19, ie white pixels with number of white neighbors)
    # Density=True calcualtes the PDF, which in the case of unit bin edges is the
    # equivalent of normalizing the values to their sum.
    with np.errstate(divide="ignore", invalid="ignore"):
        values, _ = np.histogram(
            V,
            bins=counting_bin_edges,
            range=(counting_bin_edges.min(), counting_bin_edges.max()),
            density=True,
        )
    # np.histogram gives NaN values when division by 0 occurs,
    # so check if nan in the array, if so fill nan with 0 values
    if True in np.isnan(values):
        np.nan_to_num(values, nan=0.0, copy=False)
    # use the check below if uncertain about density=True statement
    # if int(values.sum())!=1:
    #     if int(values.sum().astype(np.float16))!=1:
    #         raise RuntimeError("_ctas_mod: sum of the TAS features does not add to 1, density=True not working")
    return values


def count_pixels_tas(img: np.array) -> np.ndarray:
    """
    count_pixels_tas: takes an image and performs the pixel counting step
    in TAS feature extraction via convolution.

    INPUTS #=====================================

    img: np.array = the image whose pixel values are to be analyzed

    OUTPUTS #=====================================

    np.array = the TAS feature values for the given image

    #================================================

    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07

    ADAPTED FROM Mahotas package v: mahotas\features\tas.py
    """
    # if IMG == 2D
    if len(img.shape) == 2:
        kernel = _TAS_2Dkernel
        binedges = _TAS_binedges2D
    # if IMG == 3D
    elif len(img.shape) == 3:
        kernel = _TAS_3Dkernel
        binedges = _TAS_binedges3D
    else:
        logger.critical(
            f"count_pixels_tas: Cannot compute TAS for image of {len(img.shape)} dimensions"
        )
        raise NotImplementedError(
            f"count_pixels_tas: Cannot compute TAS for image of {len(img.shape)} dimensions"
        )
    result = _ctas_mod(bwimg=img > 0, kernel=kernel, counting_bin_edges=binedges)
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    testImg = np.zeros(shape=(20, 20, 20))
    testImg[:10, :10, :10] = 1

    testImg = np.zeros(shape=(20, 20))
    testImg[:10, :10] = 1

    count_pixels_tas(testImg)
