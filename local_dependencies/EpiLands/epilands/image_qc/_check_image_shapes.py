import logging
import os

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def check_image_shapes(image_data: dict) -> list:
    image_shapes = list(map(lambda x: x.shape, image_data.values()))
    if not all(element == image_shapes[0] for element in image_shapes):
        logger.error("Image dimensions do not match")
        logger.error(image_shapes)
        for name, shape in zip(
            image_data.keys(), list(map(lambda x: x.shape, image_data.values()))
        ):
            logger.error(f"{name}: shape {shape}")
        raise ValueError("Image dimensions do not match")
    return image_shapes
