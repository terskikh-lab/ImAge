import re
import os
import sys
from imagepubautomation.gputools import gpuinit

# Initialize GPU number
print(sys.argv[1])
gpuinit(gpuN=sys.argv[1]) #user input GPU number
# gpuinit() #randomly selected GPU instance
# gpuinit(gpuN=-1) #no GPU


from imagepubautomation.segmentation.modules import ImageObjectTransformer
from imagepubautomation.segmentation.processes import (
    normalize_segmentation_image,
    segment_images,
    resize_masks,
    resize_segmentation_image,
)


import logging

logger = logging.getLogger("imagepubautomation")


from config import (
    folder,
    experiment_name,
    dataPathRoot,
    resultsPathRoot,
    segmentation_channel,
    search_pattern,
    metadata_pattern,
    channelIndex,
    rowIndex,
    colIndex,
    zIndex,
    FOVIndex,
    tIndex,
    run_3d,
    run_illumination_correction,
    imagedims,
)


try:
    assert os.path.exists(dataPathRoot)
    assert os.path.exists(resultsPathRoot)
    loadPath = dataPathRoot / folder
    savePath = resultsPathRoot / experiment_name
    savePath.mkdir(parents=True, exist_ok=True)

    segmentation_mod = ImageObjectTransformer(
        name="segmentation",
        image_directory=loadPath,
        output_directory=savePath,
        segmentation_channel=segmentation_channel,
        search_pattern=search_pattern,
        metadata_pattern=metadata_pattern,
        channelIndex=channelIndex,
        rowIndex=rowIndex,
        colIndex=colIndex,
        zIndex=zIndex,
        FOVIndex=FOVIndex,
        tIndex=tIndex,
        run_3d=run_3d,
        run_illumination_correction=run_illumination_correction,
    )

    segmentation_mod.add_process(
        resize_segmentation_image, InitialDims=imagedims, FinalDims=(1, 1)
    )
    segmentation_mod.add_process(normalize_segmentation_image)
    segmentation_mod.add_process(segment_images)
    segmentation_mod.add_process(resize_masks, InitialDims=(1, 1), FinalDims=imagedims)
    segmentation_mod.run()
except Exception as e:
    logger.error(f"There was an error processing {experiment_name}, see below")
    logger.error(e)
