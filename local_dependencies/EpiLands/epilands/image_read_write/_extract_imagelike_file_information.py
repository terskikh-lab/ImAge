import os
import re
import logging
from typing import Union, Optional
import pandas as pd
from ._parse_imagelike_filename_metadata import parse_imagelike_filename_metadata
from ..generic_read_write import find_all_files

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def extract_imagelike_file_information(
    file_path: str,
    search_pattern: Union[str, re.Pattern],
    metadata_pattern: Union[str, re.Pattern],
    channelIndex: int,
    rowIndex: int,
    colIndex: int,
    zIndex: int,
    FOVIndex: Union[tuple, int],
    tIndex: int,
) -> pd.DataFrame:
    """
    Description load_file_paths:
    loads files into memory, saves their paths to a dict

    INPUTS #=====================================

    image_directory: str = master folder containing raw images to be analyzed. This folder may contain subfolders,
    they will all be searched.

    OUTPUTS #=====================================
    images_file_information: pd.DataFrame = dataframe containing file paths and well indexing data

    channel_info: list = list of channel names
    #================================================

    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    """
    if not os.path.isdir(file_path):
        raise ValueError(f"file_path does not exist: {file_path}")
    file_list = find_all_files(
        file_path,
        pattern=search_pattern,
    )
    logger.info(f"Found {len(file_list)} {search_pattern} files in the given directory")
    if len(file_list) == 0:
        raise ValueError(f"No image files found in directory: {file_path}")
    ## We will start by generating new names for the files
    file_information = parse_imagelike_filename_metadata(
        files=pd.Series(file_list),
        pattern=metadata_pattern,
        channelIndex=channelIndex,
        rowIndex=rowIndex,
        colIndex=colIndex,
        zIndex=zIndex,
        FOVIndex=FOVIndex,
        tIndex=tIndex,
    )
    if channelIndex is not None:
        # check if any of the channels have more/less images
        if len(file_information["channel"].value_counts().unique()) != 1:
            channelcounts = file_information["channel"].value_counts()
            raise ValueError(
                f"ERROR: MISSING OR EXTRA IMAGES. \n\nNumber of images given for one channel does not match number of images given for another channel. See Below: \n{channelcounts}"
            )
        # create a list of all channels detected
        channels = list(file_information["channel"].unique())
        # check if channels result in a non-integer division (ie, missing images)
        if (len(file_list) / len(channels)) != (len(file_list) // len(channels)):
            raise ValueError(
                "ERROR: NUMBER OF IMAGES PROVIDED DOES NOT AGREE WITH NUMBER OF CHANNELS DETECTED. \
                PLEASE CHECK INPUT IMAGES"
            )
        logger.info(f"Channels Detected: {channels}")

    return file_information


# import os
# import re
# import logging
# from typing import Union, Optional
# import pandas as pd
# from ._parse_imagelike_filename_metadata import parse_image_filename_metadata
# from ._find_all_files import find_all_files

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# # ================================================
# # Image Segmentation and Preprocessing Functions
# # ================================================
# # GLOBAL VARS:

# def extract_object_image_file_information(
#     object_image_directory: Optional[str],
#     pat: Union[str, re.Pattern] = re.compile(
#         "row([0-9]+)col([0-9]+)fov([0-9]+)"
#         ),
#     channelIndex: int = None,
#     rowIndex: int = 0,
#     colIndex: int = 1,
#     zIndex: int = None,
#     FOVIndex: int = 2,
#     tIndex: int = None
#     ) -> None:
#     segmentation_files = find_all_files(
#         path=object_image_directory, search_str="_segmented"
#     )
#     # initialize dataframe to store TAS features
#     file_information = parse_image_filename_metadata(
#         image_files=segmentation_files,
#         pat=pat,
#         channelIndex=channelIndex,
#         rowIndex=rowIndex,
#         colIndex=colIndex,
#         zIndex=zIndex,
#         FOVIndex=FOVIndex,
#         tIndex=tIndex
#     )
#     if not all(
#         [
#             i in file_information.columns
#             for i in ["WellIndex", "row", "column", "FOV"]
#         ]
#     ):
#         raise ValueError(
#             "images_file_information does not have the expected columns"
#         )
#     return file_information


# import os
# import re
# import logging
# from typing import Union, Optional
# import pandas as pd
# from ._parse_imagelike_filename_metadata import parse_image_filename_metadata
# from ._find_all_files import find_all_files

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# # ================================================
# # Image Segmentation and Preprocessing Functions
# # ================================================
# # GLOBAL VARS:

# def extract_original_image_file_information(
#     image_directory: Optional[str],
#     segmentation_channel: str,
#     pat: Union[
#         str, re.Pattern
#     ] = re.compile("r([0-9]{2})c([0-9]{2})f([0-9]{2})p([0-9]{2})-([a-zA-Z0-9]{3})"),
#     channelIndex: int = 4,
#     rowIndex: int = 0,
#     colIndex: int = 1,
#     zIndex: int = 3,
#     FOVIndex: Union[tuple, int] = 2,
#     tIndex: int = None,
# ) -> pd.DataFrame:
#     """
#     Description load_file_paths:
#     loads files into memory, saves their paths to a dict

#     INPUTS #=====================================

#     image_directory: str = master folder containing raw images to be analyzed. This folder may contain subfolders,
#     they will all be searched.

#     OUTPUTS #=====================================
#     images_file_information: pd.DataFrame = dataframe containing file paths and well indexing data

#     channel_info: list = list of channel names
#     #================================================

#     Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
#     """
#     if not os.path.isdir(image_directory):
#         raise ValueError(f"image_directory does not exist: {image_directory}")
#     file_paths = find_all_files(
#         image_directory,
#         search_str=".tif",
#     )
#     image_file_list = pd.Series(file_paths)
#     logger.info(f"Found {len(image_file_list)} .tif files in the given directory")
#     if len(image_file_list) == 0:
#         raise ValueError(f'No image files found in directory: {image_directory}')
#     ## We will start by generating new names for the files
#     images_file_information = parse_image_filename_metadata(
#         image_files=image_file_list,
#         pat=pat,
#         channelIndex=channelIndex,
#         rowIndex=rowIndex,
#         colIndex=colIndex,
#         zIndex=zIndex,
#         FOVIndex=FOVIndex,
#         tIndex=tIndex,
#     )
#     # check if any of the channels have more/less images
#     if len(images_file_information["channel"].value_counts().unique()) != 1:
#         channelcounts = images_file_information["channel"].value_counts()
#         raise ValueError(
#             f"ERROR: MISSING OR EXTRA IMAGES. \n\nNumber of images given for one channel does not match number of images given for another channel. See Below: \n{channelcounts}"
#         )

#      # create a list of all channels detected
#     channels = list(images_file_information["channel"].unique())
#     # check if channels result in a non-integer division (ie, missing images)
#     if (len(image_file_list)/len(channels)) != (len(image_file_list)//len(channels)):
#         raise ValueError(
#             "ERROR: NUMBER OF IMAGES PROVIDED DOES NOT AGREE WITH NUMBER OF CHANNELS DETECTED. \
#             PLEASE CHECK INPUT IMAGES"
#         )
#     logger.info(f"Channels Detected: {channels}")
#     if segmentation_channel not in channels:
#         raise ValueError(
#             f"segment_images: {segmentation_channel} is not a valid channel"
#             + f"\nValid channels are: {channels}"
#         )
#     if not all(
#         [
#             i in images_file_information.columns
#             for i in ["WellIndex", "channel", "row", "column", "FOV"]
#         ]
#     ):
#         raise ValueError(
#             "images_file_information does not have the expected columns"
#         )
#     return images_file_information


# import pandas as pd
# import re
# import os
# from typing import Union


# def extract_wellIdx_fov(
#     filenames: pd.Series,
#     wellIdx_fov_pattern: Union[str, re.Pattern] = re.compile(
#         "row([0-9]+)col([0-9]+)fov([0-9]+)"
#     ),
# ):
#     """Extracts a field of view from a filename given the filename is in the IC200 output file format"""
#     if not isinstance(filenames, pd.Series):
#         filenames = pd.Series(filenames).astype(str)
#     file_information = filenames.str.extract(wellIdx_fov_pattern)
#     if file_information.shape[0] == 0:
#         raise ValueError("No wellIdx_fov_pattern found in filenames")
#     file_information.columns = ["row", "col", "fov"]
#     file_information["wellindex"] = (
#         file_information["row"].astype(str)
#         + file_information["col"]
#         .astype(str)
#         .map(lambda s: "00" + s if len(s) == 1 else "0" + s)
#     ).astype(int)
#     file_information["path"] = filenames
#     file_information["filename"] = filenames.map(lambda s: os.path.split(s)[-1])
#     return file_information
