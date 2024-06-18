import os
import time
import logging

from ..classes.RunTimeCounter import RunTimeCounter

logger = logging.getLogger("MultiProcessTools")
logger.setLevel(logging.DEBUG)


def wait_until_file_exists(path: os.PathLike, timeout: int) -> bool:
    """
    waits until the file exists.
    """
    if not os.path.exists(path):
        timecounter = RunTimeCounter(timeout)
        while not os.path.exists(path):
            timecounter.check_runtime()
            logger.info(f"Waiting for {path} to be created...")
            time.sleep(5)
    return True
