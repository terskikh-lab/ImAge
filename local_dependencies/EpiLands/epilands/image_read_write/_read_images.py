import os
import numpy as np
import pandas as pd
import logging
import tifffile as tiff
from typing import Union, Dict, AnyStr
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def read_images(
    image_files: Union[list, np.ndarray, pd.Series], return_3d: bool = False
):
    if not isinstance(image_files, pd.Series):
        image_files = pd.Series(image_files)
    image_files = image_files.sort_values(
        ascending=True
    )  ### Generalize regex for usage using plane -> int sorting
    tmpImg = tiff.imread(files=image_files.to_list())
    if len(image_files) > 1 and return_3d == False:
        tmpImg = tmpImg.max(axis=0)
    return tmpImg


def load_images(
    file_information: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    with ThreadPoolExecutor() as executor:
        image_data = {
            channel: executor.submit(read_images, data["file_path"])
            for channel, data in file_information.groupby(["channel"])
        }
    for image in image_data:
        image_data[image] = image_data[image].result()
    return image_data
