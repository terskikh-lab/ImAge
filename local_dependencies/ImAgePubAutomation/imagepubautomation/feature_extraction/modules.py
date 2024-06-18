from __future__ import annotations
import os
import logging
import pandas as pd
from tqdm import tqdm, trange
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Union, Optional, Callable, List, Tuple, Dict, Any, Union, Optional

from multiprocesspipelines.module import Module

from epilands.generic_read_write import (
    save_dataframe_to_csv,
    save_dataframe_to_h5_file,
)
from epilands.image_read_write import (
    extract_imagelike_file_information,
    read_segmentation_data,
)
from epilands.config import ALL_SUBDIRS

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


class ObjectInput(Module):
    def __init__(
        self,
        name: str,
        object_image_directory: str,
        output_directory: str,
        search_pattern: Union[str, re.Pattern] = ".hdf5",
        metadata_pattern: Union[str, re.Pattern] = re.compile(
            "row([0-9]+)col([0-9]+)fov([0-9]+)"
        ),
        channelIndex: int = None,
        rowIndex: int = 0,
        colIndex: int = 1,
        zIndex: int = None,
        FOVIndex: int = 2,
        tIndex: int = None,
    ):
        super().__init__(
            name=name,
            output_directory=output_directory,
            loggers=["MultiProcessTools", *ALL_SUBDIRS],
        )
        if not os.path.isdir(object_image_directory):
            self.create_directory(object_image_directory)
            object_image_directory = self.get_directory(object_image_directory)
        self.file_information = extract_imagelike_file_information(
            file_path=object_image_directory,
            search_pattern=search_pattern,
            metadata_pattern=metadata_pattern,
            channelIndex=channelIndex,
            rowIndex=rowIndex,
            colIndex=colIndex,
            zIndex=zIndex,
            FOVIndex=FOVIndex,
            tIndex=tIndex,
        )
        self.file_information_grouped = self.file_information.sample(
            frac=1, replace=False
        ).groupby(["row", "column", "FOV"])

    def _load_segmentation_result(self, segmentation_filepath):
        (
            self.objects,
            self.details,
        ) = read_segmentation_data(segmentation_filepath)


class ObjectFeatureTransformer(ObjectInput):
    def _create_feature_dataframe(self, wellIdx, field_of_view):
        # create empty dataframe to append data to, indexed by cell number
        # add in the relevant data for wellindex, FOV, x&y coordinates, object area, morphology
        df_feature_data = pd.DataFrame(index=self.objects.keys())
        df_feature_data["WellIndex"] = int(wellIdx)
        df_feature_data["FieldOfView"] = int(field_of_view)
        if self.details["points"].shape[1] == 2:
            df_feature_data["XCoord"] = self.details["points"][:, 0]
            df_feature_data["YCoord"] = self.details["points"][:, 1]
        elif self.details["points"].shape[1] == 3:
            df_feature_data["ZCoord"] = self.details["points"][:, 0]
            df_feature_data["XCoord"] = self.details["points"][:, 1]
            df_feature_data["YCoord"] = self.details["points"][:, 2]
        else:
            raise NotImplementedError(
                f"details['points'].shape is not 2D or 3D but \
                {self.details['points'].shape[1]}D, what is the dimension of your images??"
            )
        df_feature_data["STARDIST_probability"] = self.details["prob"]
        self.df_feature_data = df_feature_data

    def _append_feature_data(self):
        # add the TAS data to the dataframe
        self.df_feature_data = self.df_feature_data.merge(
            pd.DataFrame.from_dict(self.features, orient="index"),
            left_index=True,
            right_index=True,
        )

    def run(self):
        logger.info(f"Running Feature Extraction")
        # initialize run information
        # segmentation_output_directory = self.directories['segmentation_output_directory']
        try:
            self.create_directory("feature_extraction")
            output_directory = self.get_directory("feature_extraction")
            for row_col_fov, group_data in tqdm(
                self.file_information_grouped, "Progress extracting features:"
            ):
                # Create a new name to save the segmenetation results for this set of images
                final_file_name = (
                    group_data["filename"].iloc[0].replace("_segmented", "_features")
                )
                temp_file_name = final_file_name.replace("hdf5", "tmp")
                file_not_in_use = self.create_temp_file(
                    final_file_name=final_file_name,
                    temp_file_name=temp_file_name,
                    path="feature_extraction",
                )
                if file_not_in_use == False:
                    logger.debug(f"File {temp_file_name} already exists, skipping...")
                    continue
                group_name = (
                    f"row{row_col_fov[0]}col{row_col_fov[1]}fov{row_col_fov[2]}"
                )
                # extract wellindex and FOV, check if it matches segmentation images. Raise error if not
                logger.info(f"Analyzing data from image: {group_name}....")

                self._load_segmentation_result(group_data["file_path"].iloc[0])

                if len(self.objects) == 0:
                    logger.warning(f"Image {group_name} had no segmented objects")
                    continue

                self._create_feature_dataframe(
                    wellIdx=group_data["WellIndex"].iloc[0],
                    field_of_view=group_data["FOV"].iloc[0],
                )

                self.run_all_processes()

                self._append_feature_data()
                # np.seterr(invalid="warn")
                save_dataframe_to_h5_file(
                    os.path.join(output_directory, final_file_name),
                    self.df_feature_data,
                )
                self.delete_tempfile(
                    os.path.join(
                        self.get_directory("feature_extraction"), temp_file_name
                    )
                )
            self.cleanup()
        except Exception as e:
            logger.exception(e)
            logger.error("An exception occurred, cleaning up...")
            self.cleanup()

    def run_multiprocessing(self, process):
        logger.info(f"Running Feature Extraction")
        # initialize run information
        # segmentation_output_directory = self.directories['segmentation_output_directory']
        try:
            self.create_directory("feature_extraction")

            def _run_multiprocessing(group_data):
                final_file_name = (
                    group_data["filename"].iloc[0].replace("_segmented", "_features")
                )
                temp_file_name = final_file_name.replace("hdf5", "tmp")
                file_not_in_use = self.create_temp_file(
                    final_file_name=final_file_name,
                    temp_file_name=temp_file_name,
                    path="feature_extraction",
                )
                if file_not_in_use == False:
                    logger.debug(f"File {temp_file_name} already exists, skipping...")
                    return
                process(
                    output_directory=self.get_directory("feature_extraction"),
                    group_data=group_data,
                )
                self.delete_tempfile(
                    os.path.join(
                        self.get_directory("feature_extraction"), temp_file_name
                    )
                )

            with ProcessPoolExecutor() as executor:
                # Create a new name to save the segmenetation results for this set of images
                futures = executor.map(
                    _run_multiprocessing,
                    [
                        group_data
                        for row_col_fov, group_data in self.file_information_grouped
                    ],
                )
                for future in tqdm(
                    as_completed(futures), "Progress extracting features:"
                ):
                    pass
                executor.shutdown(wait=True)
            self.cleanup()
        except Exception as e:
            logger.exception(e)
            logger.error("An exception occurred, cleaning up...")
            self.cleanup()
