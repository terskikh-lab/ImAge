import numpy as np
from .tags import (
    receives_image_data,
    receives_masks_objects_details,
    receives_segmentation_image,
    outputs_segmentation_image,
    outputs_masks_objects_details,
    outputs_image_data,
)
from epilands.image_segmentation import (
    segment_image_stardist2d,
    segment_image_stardist3d,
)
from epilands.image_preprocessing import glaylconvert, ezresizeimg, ezresizemask


@receives_segmentation_image
@outputs_segmentation_image
def normalize_segmentation_image(segmentation_image):
    return glaylconvert(
        img=segmentation_image,
        orgLow=np.percentile(segmentation_image, 1),
        orgHigh=np.percentile(segmentation_image, 99),
        qLow=0,
        qHigh=1,
    )


@receives_segmentation_image
@outputs_masks_objects_details
def segment_images(segmentation_image):
    if len(segmentation_image.shape) == 2:
        return segment_image_stardist2d(segmentation_image)
    elif len(segmentation_image.shape) == 3:
        return segment_image_stardist3d(segmentation_image)
    else:
        raise ValueError(
            f"Segmentation image must be 2d or 3d but {segmentation_image.shape} was given"
        )


@receives_segmentation_image
@outputs_segmentation_image
def normalize_segmentation_image(segmentation_image):
    return glaylconvert(
        img=segmentation_image,
        orgLow=np.percentile(segmentation_image, 1),
        orgHigh=np.percentile(segmentation_image, 99),
        qLow=0,
        qHigh=1,
    )


@receives_segmentation_image
@outputs_segmentation_image
def resize_segmentation_image(segmentation_image, InitialDims, FinalDims):
    return ezresizeimg(
        segmentation_image,
        InitialDims=InitialDims,
        FinalDims=FinalDims,
    )


@receives_masks_objects_details
@outputs_masks_objects_details
def resize_masks(masks_objects_details, InitialDims, FinalDims):
    masks = ezresizemask(
        masks_objects_details[0],
        InitialDims=InitialDims,
        FinalDims=FinalDims,
    )
    masks_objects_details = (masks, masks_objects_details[1], masks_objects_details[2])
    return masks_objects_details


@receives_image_data
@outputs_image_data
def resize_image_data(image_data, InitialDims, FinalDims):
    for key, data in image_data.items():
        image_data[key] = ezresizeimg(
            data,
            InitialDims=InitialDims,
            FinalDims=FinalDims,
        )
    return image_data
