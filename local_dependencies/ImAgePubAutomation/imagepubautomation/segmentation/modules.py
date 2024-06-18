from __future__ import annotations
import os
import logging
import time
import numpy as np
import pandas as pd
from beartype import beartype
from tqdm import tqdm, trange
import basicpy
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Union, Optional, Callable, List, Tuple, Dict, Any, Union, Optional
from functools import partial

from multiprocesstools import (
    RunTimeCounter,
    wait_until_file_exists,
    run_func_IO_loop,
)
from multiprocesspipelines.module import Module
from epilands.image_qc import (
    check_image_shapes,
    power_spectrum_loglog_slope,
    percent_max,
    percent_median2SD,
)
from epilands.tools import (
    join_tuple,
)
from epilands.image_read_write import (
    extract_imagelike_file_information,
    save_segmentation_data,
    read_images,
)
from epilands.generic_read_write import (
    ezload,
    ezsave,
    save_dataframe_to_csv,
)

from epilands.config import ALL_SUBDIRS

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


class ImageInput(Module):
    def __init__(
        self,
        name: str,
        image_directory: str,
        output_directory: str,
        search_pattern: Union[str, re.Pattern] = ".tiff",
        metadata_pattern: Union[str, re.Pattern] = re.compile(
            "r([0-9]{2})c([0-9]{2})f([0-9]{2})p([0-9]{2})-([a-zA-Z0-9]{3})"
        ),
        channelIndex: int = 4,
        rowIndex: int = 0,
        colIndex: int = 1,
        zIndex: int = 3,
        FOVIndex: Union[tuple, int] = 2,
        tIndex: int = None,
        run_3d: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            output_directory=output_directory,
            loggers=[sub_package_name, "MultiProcessTools", *ALL_SUBDIRS],
        )

        ## We will start by generating new names for the files
        image_file_information = extract_imagelike_file_information(
            file_path=image_directory,
            search_pattern=search_pattern,
            metadata_pattern=metadata_pattern,
            channelIndex=channelIndex,
            rowIndex=rowIndex,
            colIndex=colIndex,
            zIndex=zIndex,
            FOVIndex=FOVIndex,
            tIndex=tIndex,
        )

        # create a list of all channels detected
        self._run_3d = run_3d
        self.num_wells = image_file_information["WellIndex"].nunique()
        self.channels = list(image_file_information.channel.unique())
        self.image_file_information = image_file_information
        self.image_file_information_grouped = image_file_information.sample(
            frac=1, replace=False
        ).groupby(["row", "column", "FOV"], sort=False)

        logger.info(f"run3D: {self._run_3d}")
        logger.info(f"num_wells: {self.num_wells}")
        logger.info(f"channels: {self.channels}")

    @property
    def run_3d(self):
        return self._run_3d

    def _load_group_image_data(self, group_file_information):
        logger.debug("started o2_load_group_image_data...")
        ## Read in the images for the given row, column, and FOV
        # read in the image data
        channels_files = {
            channel: data["file_path"]
            for channel, data in group_file_information.groupby(["channel"])
        }
        _read_images = partial(read_images, return_3d=self.run_3d)
        with ThreadPoolExecutor() as executor:
            futures = executor.map(_read_images, [f for _, f in channels_files.items()])
            image_data = {
                key: future for key, future in zip(channels_files.keys(), futures)
            }
            executor.shutdown(wait=True)
        # for image in image_data:
        #     image_data[image] = image_data[image].result()
        # Check images to make sure the proper files are being used
        image_shapes = check_image_shapes(image_data)
        logger.debug(f"self.image_data: {image_data}")
        logger.debug(f"image_shapes: {image_shapes}")
        self.image_data = image_data
        logger.debug("finished o2_load_group_image_data...")


class ImageObjectTransformer(ImageInput):
    # define class variables
    timeout = 60 * 25  # 25 mins
    illumination_correction_min_num_wells = 25

    def __init__(
        self,
        name,
        image_directory: str,
        output_directory: str,
        segmentation_channel: str,
        search_pattern: Union[str, re.Pattern] = ".tiff",
        metadata_pattern: Union[str, re.Pattern] = re.compile(
            "r([0-9]{2})c([0-9]{2})f([0-9]{2})p([0-9]{2})-([a-zA-Z0-9]{3})"
        ),
        channelIndex: int = 4,
        rowIndex: int = 0,
        colIndex: int = 1,
        zIndex: int = 3,
        FOVIndex: Union[tuple, int] = 2,
        tIndex: int = None,
        run_3d: bool = False,
        run_illumination_correction: bool = True,
    ) -> None:
        """Initialize the channel to segment on and create output directories"""
        time.sleep(np.random.random())
        # Initialize variables, type checks, and create output directories

        super().__init__(
            name=name,
            image_directory=image_directory,
            output_directory=output_directory,
            search_pattern=search_pattern,
            metadata_pattern=metadata_pattern,
            channelIndex=channelIndex,
            rowIndex=rowIndex,
            colIndex=colIndex,
            zIndex=zIndex,
            FOVIndex=FOVIndex,
            tIndex=tIndex,
            run_3d=run_3d,
        )

        self._segmentation_channel = segmentation_channel
        self._run_illumination_correction = run_illumination_correction
        logger.info(f"segmentation_channel: {self._segmentation_channel}")
        logger.info(f"run_illumination_correction: {self._run_illumination_correction}")

    @property
    def run_illumination_correction(self):
        return self._run_illumination_correction

    @property
    def segmentation_channel(self):
        return self._segmentation_channel

    def update_segmentation_channel(self, segmentation_channel):
        self._segmentation_channel = segmentation_channel

    def update_run_illumination_correction(self, run_illumination_correction: bool):
        self._run_illumination_correction = run_illumination_correction

    def _calculate_illumination_correction_fields(
        self,
    ) -> None:
        self.create_directory("illumination_correction_models")
        output_directory = self.get_directory("illumination_correction_models")
        if self.run_illumination_correction == False:
            logger.error(
                "_calculate_illumination_correction_fields was called but run_illumination_correction is false"
            )
            raise ValueError(
                "_calculate_illumination_correction_fields was called but run_illumination_correction is false"
            )
        # Check if there are enough wells to perform illumination correction
        if self.num_wells < self.illumination_correction_min_num_wells:
            logger.warning(
                f"There are only {self.num_wells} wells in this experiment \
                    therefore, illumination correction cannot be accurately performed"
            )
            logger.warning("Running without illumination correction")
            self.illumination_correction_filenames = None
            self.illumination_correction_models = None
            return

        self.illumination_correction_filenames = {}
        self.illumination_correction_models = {}
        image_illumination_groups = self.image_file_information.sample(
            frac=1, replace=False
        ).groupby(["channel", "FOV"], sort=False)
        for group, data in tqdm(image_illumination_groups):
            group_name = join_tuple(group)
            final_file_name = f"{group_name}_illumination_correction_model.pickle"
            temp_file_name = final_file_name.replace(".pickle", ".tmp")
            self.illumination_correction_filenames[group] = os.path.join(
                output_directory, final_file_name
            )
            file_not_in_use = self.create_temp_file(
                final_file_name=final_file_name,
                temp_file_name=temp_file_name,
                path=output_directory,
            )
            if file_not_in_use == False:
                logger.info(f"{group_name} already in progress, skipping...")
                continue
            # Check if the files already exist. If they do, skip the calculation.
            # If they don't, calculate the flatfield and darkfield.
            # If only one exists, log and error and delete it.
            logger.info(f"Processing {group_name}")
            fov_filepaths = {
                str(rowcol): tmpDf["file_path"]
                for rowcol, tmpDf in data.groupby(["row", "column"])
            }
            # logger.debug(f"WellIdxs seeing correction all at once:\n {fov_filepaths.keys()}")
            # logger.debug(f"Files seeing correction all at once:\n {pd.concat([*fov_filepaths.values()]).str.split("/").apply(lambda x: x[-1])}")
            if self.run_3d == True:
                msgfovs = (
                    pd.concat([*fov_filepaths.values()])
                    .str.extract("r[0-9a-zA-Z]+c[0-9a-zA-Z]+(f[0-9a-zA-Z]+)")[0]
                    .unique()
                )
                logger.debug(f"FOVS seeing correction all at once:\n {msgfovs}")

            _read_images = partial(read_images, return_3d=self.run_3d)
            with ThreadPoolExecutor() as executor:
                futures = executor.map(_read_images, fov_filepaths.values())
                executor.shutdown(wait=True)
            group_images = np.array([future for future in futures])
            correction_model = basicpy.BaSiC()
            correction_model.get_darkfield = True
            correction_model.fit(group_images)
            self.illumination_correction_models[group] = correction_model
            ezsave(
                {"model": correction_model, "dummy": []},
                self.illumination_correction_filenames[group],
            )
            logger.info(
                f"Saved model to {self.illumination_correction_filenames[group]}"
            )
            self.delete_tempfile(os.path.join(output_directory, temp_file_name))

    def _correct_illumination(
        self,
        image: np.ndarray,
        channel: str,
        field_of_view: str,
    ) -> np.ndarray:
        if self.run_illumination_correction == False:
            logger.error(
                "_correct_illumination was called but run_illumination_correction is false"
            )
            raise ValueError(
                "_correct_illumination was called but run_illumination_correction is false"
            )
        logger.info(f"illumination correcting image data for channel {channel}")
        # if the model is not already loaded, load it. If it doesn't exist,
        # wait until it is created. If timeout time is reached, raise an error.
        # If the mod
        if (channel, field_of_view) in self.illumination_correction_models:
            correction_model = self.illumination_correction_models[
                (channel, field_of_view)
            ]
        else:
            wait_until_file_exists(
                path=self.illumination_correction_filenames[(channel, field_of_view)],
                timeout=self.timeout,
            )
            time.sleep(np.random.rand())
            # correction_model = basicpy.BaSiC.load_model(self.illumination_correction_filenames[channel, field_of_view])
            correction_model = run_func_IO_loop(
                func=ezload,
                func_args={
                    "file": self.illumination_correction_filenames[
                        (channel, field_of_view)
                    ]
                },
                timeout=self.timeout,
            )
            correction_model = correction_model["model"]
            self.illumination_correction_models[(channel, field_of_view)] = (
                correction_model
            )
        corrected_image = correction_model.transform(image)[0]
        return corrected_image

    def _apply_illumination_correction(self, field_of_view):
        if self.run_illumination_correction == False:
            logger.error(
                "_apply_illumination_correction was called but run_illumination_correction is false"
            )
            raise ValueError(
                "_apply_illumination_correction was called but run_illumination_correction is false"
            )
        logger.debug("started o3_image_data_illumination_correction...")
        ## preprocess the image
        # Check if illumination correction file names exist
        try:
            for channel in self.image_data.keys():
                logger.debug(f"self.image data: {self.image_data}")
                tmpImg = self.image_data[channel]
                # Set the image data to the corrected image data
                self.image_data[channel] = self._correct_illumination(
                    image=tmpImg,
                    channel=channel,
                    field_of_view=field_of_view,
                )
        except AttributeError as e:
            logger.error("Must run o1_calculate_illumination_correction_fields first")
            raise e
        logger.debug("finished o3_image_data_illumination_correction...")

    def _select_segmentation_image(self):
        self.segmentation_image = self.image_data[self.segmentation_channel].copy()

    def run(self):
        try:
            self.create_directory("segmentation")
            output_directory = self.get_directory("segmentation")
            self.segmentation_files = []
            if self.run_illumination_correction == True:
                # Calculate the illumination correction fields for each channel/fov
                self._calculate_illumination_correction_fields()
            # iterate through every file in the given file dict/segmentation channel pair to segment each set of images
            for row_col_fov, group_file_information in tqdm(
                self.image_file_information_grouped,
                f"Progress segmenting images: ",
            ):
                final_file_name = f"row{row_col_fov[0]}col{row_col_fov[1]}fov{row_col_fov[2]}_segmented.hdf5"
                temp_file_name = final_file_name.replace(".hdf5", ".tmp")
                file_not_in_use = self.create_temp_file(
                    final_file_name=final_file_name,
                    temp_file_name=temp_file_name,
                    path=output_directory,
                )
                if file_not_in_use == False:
                    logger.info(f"File {temp_file_name} already exists, skipping...")
                    continue
                logger.info(f"Started {final_file_name}")
                self._load_group_image_data(
                    group_file_information=group_file_information
                )
                self._select_segmentation_image()
                if self.run_illumination_correction == True:
                    self._apply_illumination_correction(field_of_view=row_col_fov[2])

                self.run_all_processes()

                if len(self.masks_objects_details[1]) == 0:
                    logger.info(f"Well filename had no segmented objects")
                    continue

                save_segmentation_data(
                    path=output_directory,
                    filename=final_file_name,
                    image_data=self.image_data,
                    masks=self.masks_objects_details[0],
                    details=self.masks_objects_details[2],
                    objects=self.masks_objects_details[1],
                )
                self.delete_tempfile(
                    tempfile=os.path.join(output_directory, temp_file_name)
                )
                logger.info(f"Finished {final_file_name}...")
            # finished, save the dictionaries which give easy access to the .h5 files created
            logger.info("Segmentation Done!")
            self.cleanup()
            return self
        except Exception as e:
            logger.error("An exception occurred, cleaning up...")
            logger.error(e)
            self.cleanup()
            raise e
