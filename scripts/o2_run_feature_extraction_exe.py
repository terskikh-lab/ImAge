import re
import logging
from imagepubautomation.feature_extraction.modules import ObjectFeatureTransformer
from imagepubautomation.feature_extraction.processes import (
    extract_features,
)
from .config import resultsPathRoot, experiment_name

assert resultsPathRoot.exists()
logger = logging.getLogger("imagepubautomation")

experiment_output_folder = resultsPathRoot / experiment_name

try:
    feature_extraction_mod = ObjectFeatureTransformer(
        name="feature_extraction",
        object_image_directory="segmentation",
        output_directory=experiment_output_folder,
        search_pattern=".hdf5",
        metadata_pattern=re.compile("row([0-9]+)col([0-9]+)fov([0-9]+)"),
        channelIndex=None,
        rowIndex=0,
        colIndex=1,
        zIndex=None,
        FOVIndex=2,
        tIndex=None,
    )

    feature_extraction_mod.add_process(extract_features)
    feature_extraction_mod.run()


except Exception as e:
    logger.error(f"There was an error processing {experiment_name}, see below")
    logger.error(e)
