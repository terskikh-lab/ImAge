import logging
import os
from typing import Dict
from numpy import ndarray
from concurrent.futures import ThreadPoolExecutor
from ._read_images import read_images
from ..image_qc import check_image_shapes

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def load_group_images(group_file_information) -> Dict[str, ndarray]:
    logger.debug("started load_group_image_data...")
    ## Read in the images for the given row, column, and FOV
    # read in the image data
    with ThreadPoolExecutor() as executor:
        image_data = {
            channel: executor.submit(read_images, data["file_path"])
            for channel, data in group_file_information.groupby(["channel"])
        }
    for image in image_data:
        image_data[image] = image_data[image].result()
    image_data = image_data
    # Check images to make sure the proper files are being used
    image_shapes = check_image_shapes(image_data)
    logger.debug(f"image_data: {image_data}")
    logger.debug(f"image_shapes: {image_shapes}")
    logger.debug("finished load_group_image_data...")
