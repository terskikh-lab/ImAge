import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_script_output_dirs(savePathRoot, script_name):
    output_dir = os.path.join(savePathRoot, script_name)
    if os.path.exists(output_dir):
        logger.info(f"{output_dir} already exists")
        if len(os.listdir(output_dir)) > 0:
            logger.warning(f"{output_dir} contains files. These may be overwritten")
    else:
        logger.info(f"Creating {output_dir}")
        os.makedirs(output_dir, exist_ok=False)
    return output_dir
