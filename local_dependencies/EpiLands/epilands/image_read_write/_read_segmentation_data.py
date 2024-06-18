import warnings
import pandas as pd
import h5py
from typing import Union, Tuple
import numpy as np


def read_segmentation_data(
    segmentationFile,
) -> Tuple[dict, dict, list]:
    with h5py.File(segmentationFile, "r") as segmentationFile:
        details = {}
        for detail in segmentationFile["details"].keys():
            details[detail] = np.array(segmentationFile["details"][detail][:])
        # read in the image data
        objects = {
            objIdx:{
                ch:img[:] for ch, img in objSubGrp.items()
                }
            for objIdx, objSubGrp in segmentationFile["objects"].items()
        }
    return objects, details










# def _read_all_datasets_from_group_h5py(
#     group, numerical_index: bool = True, as_series: bool = False
# ) -> Union[pd.DataFrame, pd.Series]:
#     """
#     Reads all datasets from a group in a h5py file
#     """
#     if len(group.keys()) < 2:
#         warnings.warn(
#             "WARNING: Only 1 object detected in image."
#             + f"group {group.name}, keys: {group.keys()}"
#             + " Object will be kept and analyzed."
#             + " Check original image for quality control"
#         )
#     df = pd.Series(
#         [group[key][:] for key in list(group.keys())], index=list(group.keys())
#     ).to_frame()
#     if numerical_index == True:
#         df.index = df.index.astype(int)
#         df.sort_index(inplace=True)
#     if as_series == True:
#         df = df.iloc[:, 0]
#     return df


# def read_segmentation_data(
#     segmentationFile,
# ) -> Tuple[dict, pd.Series, dict, list, list]:
#     with h5py.File(segmentationFile, "r") as segmentationFile:
#         details = {}
#         for detail in segmentationFile["details"].keys():
#             details[detail] = np.array(segmentationFile["details"][detail][:])
#         # Construct objects linspace since size thresholding has not been done yet
#         objects = [*map(int, segmentationFile["objects"]["masks"].keys())]
#         masks = _read_all_datasets_from_group_h5py(
#             segmentationFile["objects"]["masks"],
#             numerical_index=True,
#             as_series=True,
#         )
#         # get a list of all the channels
#         channels = list(segmentationFile["objects"].keys())
#         # remove the masks from channels
#         channels.remove("masks")
#         # read in the image data
#         objects = {
#             ch: _read_all_datasets_from_group_h5py(
#                 segmentationFile["objects"][ch], numerical_index=True, as_series=True
#             )
#             for ch in channels
#         }
#     return objects, masks, details, objects, channels



