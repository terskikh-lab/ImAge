import os
import time
from matplotlib.figure import Figure
from typing import Union, Iterable
import logging

from ..config import DEFAULT_DPI, DEFAULT_IMAGE_FORMATS

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def save_matplotlib_figure(
    fig: Figure,
    path: str,
    filename: str,
    image_formats: list = DEFAULT_IMAGE_FORMATS,
    dpi: int = DEFAULT_DPI,
    bbox_inches: str = "tight",
    metadata=None,
    pad_inches: Union[float, int] = 0.1,
    facecolor: str = "auto",
    edgecolor: str = "auto",
    **kwargs,
) -> None:
    try:
        if not isinstance(image_formats, Iterable):
            logger.warning("Image formats must be an iterable. Checking if str...")
            if isinstance(image_formats, str):
                image_formats = [image_formats]
            else:
                raise ValueError("image_formats must be a string or a list of strings")
        for _format in image_formats:
            name = os.path.join(path, ".".join([filename, _format]))
            if os.path.exists(name):
                logger.warning(f"{name} already exists, overwriting...")
                os.remove(name)
                # logger.warning(f"{name} already exists, appending time...")
                # name = os.path.join(path, '.'.join([filename+str(time.time()),_format]))
            fig.savefig(
                fname=name,
                format=_format,
                dpi=dpi,
                bbox_inches=bbox_inches,
                metadata=metadata,
                pad_inches=pad_inches,
                facecolor=facecolor,
                edgecolor=edgecolor,
                **kwargs,
            )
            logger.info(f"Figure saved as {name}")
    except:
        logger.error(f"Could not save {filename} to {path}")
