import numpy as np
import os
import h5py
from typing import Union

from ..image_preprocessing import object_cookie_cutter


def save_segmentation_data(
    path: str,
    filename: str,
    image_data: dict,
    masks: np.ndarray,
    details: dict,
    objects: Union[list, np.ndarray],
):
    """
    STRUCTURE:
        file.hdf5
        - "details" [GROUP]
            - "detail_name" [DATASET]
                data corresponding to this detail
        - "fullmask" [GROUP]
            - "masks" [DATASET]
                image masks output from stardist (same shape as original image)
        - "objects" [GROUP]
            - "1" [GROUP]
                - "masks" [DATASET]
                    cropped mask of object 1
                - "ch1" [DATASET]
                    cropped channel 1 image of object 1
                - "ch2" [DATASET]
                    cropped channel 2 image of object 1
                - ...
                - "chN" [DATASET]
                    cropped channel N image of object 1
            - "2" [GROUP]
                - "masks" [DATASET]
                    cropped mask of object 2
                - "ch1" [DATASET]
                    cropped channel 1 image of object 2
                - "ch2" [DATASET]
                    cropped channel 2 image of object 2
                - ...
                - "chN" [DATASET]
                    cropped channel N image of object 2
            - ...
            - "N" [GROUP]
                - "masks" [DATASET]
                    cropped mask of object N
                - "ch1" [DATASET]
                    cropped channel 1 image of object N
                - "ch2" [DATASET]
                    cropped channel 2 image of object N
                - ...
                - "chN" [DATASET]
                    cropped channel N image of object N

    """
    # Create the file, open it and write the data to it
    with h5py.File(os.path.join(path, filename), "x") as hdf5_file:
        # Create group for details
        details_group = hdf5_file.create_group("details")
        # Loop the details and save each to the details group
        for detail in details.keys():
            details_group.create_dataset(name=detail, data=details[detail])
        # Full mask group
        fullmask_group = hdf5_file.create_group("fullmask")
        fullmask_group.create_dataset("masks", data=masks, dtype=masks.dtype)
        data = {}
        for channel, img in image_data.items():
            object_images, object_masks = object_cookie_cutter(
                image=img,
                mask=masks,
                objects=objects,
            )
            data[channel] = object_images
        data["masks"] = object_masks
        # Create group to store all the objects
        objects_group = hdf5_file.create_group("objects")
        for object_idx in objects:
            objects_idx_group = objects_group.create_group(str(object_idx))
            for ch in data:
                objects_idx_group.create_dataset(ch, data=data[ch][object_idx])


# def save_segmentation_data(
#     path: str,
#     filename: str,
#     image_data: dict,
#     masks: np.ndarray,
#     details: dict,
#     objects: Union[list, np.ndarray],
# ):
#     # Create the file, open it and write the data to it
#     with h5py.File(os.path.join(path, filename), "x") as hdf5_file:
#         # Create group for details
#         details_group = hdf5_file.create_group("details")
#         # Full mask group
#         fullmask_group = hdf5_file.create_group("fullmask")
#         fullmask_group.create_dataset("masks", data=masks, dtype=masks.dtype)
#         # Create group for image data
#         image_data_group = hdf5_file.create_group("image_data")
#         # Loop the details and save each to the details group
#         for detail in details.keys():
#             details_group.create_dataset(name=detail, data=details[detail])
#         channels = list(image_data.keys())
#         channel_group = image_data_group.create_group(name=channels[0])
#         masks_group = image_data_group.create_group(name="masks")
#         object_slices, object_masks = object_cookie_cutter(
#             image=image_data[channels[0]],
#             mask=masks,
#             objects=objects,
#         )
#         for object_idx, object_slice in object_slices.items():
#             channel_group.create_dataset(str(object_idx), data=object_slice)
#             masks_group.create_dataset(str(object_idx), data=object_masks[object_idx])
#         # loop through each channel, save each to the image data group
#         for channel in channels[1:]:
#             channel_group = image_data_group.create_group(name=channel)
#             object_slices, object_masks = object_cookie_cutter(
#                 image=image_data[channel],
#                 mask=masks,
#                 objects=objects,
#             )
#             for object_idx, object_slice in object_slices.items():
#                 channel_group.create_dataset(str(object_idx), data=object_slice)
