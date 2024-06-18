from typing import Dict, OrderedDict
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
from epilands.object_feature_extraction import (
    calculate_TAS_features,
    calculate_intensity_features,
    object_pixel_count,
)
from epilands.image_read_write import read_segmentation_data
from epilands.generic_read_write import save_dataframe_to_h5_file

from .tags import (
    receives_objects,
    receives_details,
    receives_features,
    outputs_objects,
    outputs_details,
    outputs_features,
)


def _all_features(object_imgs: Dict[str, np.ndarray]):
    return pd.concat(
        [
            object_pixel_count(object_imgs),
            calculate_intensity_features(object_imgs),
            calculate_TAS_features(object_imgs),
        ]
    )


@receives_objects
@outputs_features
def extract_features(objects: Dict[str, Dict[str, np.ndarray]]):
    # np.seterr(invalid="ignore")
    # Thread pool to run the feature extraction on each object,
    # for each channel.
    # Threads are made for each individual object -- could
    # be made more efficient by running ProcessPoolExecutor with chunking
    # also add np.seterr(invalid='ignore') to ignore divide by zero errors
    # Caused by the fact that some objects have no intensity data for a channel
    with ThreadPoolExecutor() as executor:
        futures = executor.map(_all_features, [img for _, img in objects.items()])
        executor.shutdown(wait=True, cancel_futures=False)
    objects_features = {
        objIdx: future for objIdx, future in zip(objects.keys(), futures)
    }
    return objects_features


def create_feature_dataframe(objects, details, wellIdx, field_of_view):
    # create empty dataframe to append data to, indexed by cell number
    # add in the relevant data for wellindex, FOV, x&y coordinates, object area, morphology
    df_feature_data = pd.DataFrame(index=objects.keys())
    df_feature_data["WellIndex"] = int(wellIdx)
    df_feature_data["FieldOfView"] = int(field_of_view)
    if details["points"].shape[1] == 2:
        df_feature_data["XCoord"] = details["points"][:, 0]
        df_feature_data["YCoord"] = details["points"][:, 1]
    elif details["points"].shape[1] == 3:
        df_feature_data["ZCoord"] = details["points"][:, 0]
        df_feature_data["XCoord"] = details["points"][:, 1]
        df_feature_data["YCoord"] = details["points"][:, 2]
    else:
        raise NotImplementedError(
            f"details['points'].shape is not 2D or 3D but \
            {details['points'].shape[1]}D, what is the dimension of your images??"
        )
    df_feature_data["STARDIST_probability"] = details["prob"]
    return df_feature_data


def append_feature_data(df_feature_data, features):
    # add the TAS data to the dataframe
    return df_feature_data.merge(
        pd.DataFrame.from_dict(features, orient="index"),
        left_index=True,
        right_index=True,
    )


def devmultiprocessing_feature_extraction(output_directory, group_data):
    final_file_name = group_data["filename"].iloc[0].replace("_segmented", "_features")
    segmentation_filepath = group_data["file_path"].iloc[0]
    wellIdx = group_data["WellIndex"].iloc[0]
    field_of_view = group_data["FOV"].iloc[0]

    objects, details = read_segmentation_data(segmentation_filepath)
    df_feature_data = create_feature_dataframe(objects, details, wellIdx, field_of_view)
    features = extract_features(objects)
    df_full_data = append_feature_data(df_feature_data, features)
    save_dataframe_to_h5_file(
        os.path.join(output_directory, final_file_name),
        df_full_data,
    )
