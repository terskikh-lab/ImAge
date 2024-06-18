import time
import logging
from types import FunctionType
import numpy as np

from ..classes.RunTimeCounter import RunTimeCounter

logger = logging.getLogger("MultiProcessTools")
logger.setLevel(logging.DEBUG)


## We should turn this into a decorator
def run_func_IO_loop(func: FunctionType, func_args: dict, timeout: float):
    """
    Runs a function in a loop until it returns with no IO error.
    """
    busy = True
    timecounter = RunTimeCounter(timeout)
    while busy == True:
        try:
            return_value = func(**func_args)
            busy = False
        except Exception as e:
            logger.error(f"Couldn't run {func.__name__}: " + str(e))
        time.sleep(np.random.uniform(0.01, 0.5))
        timecounter.check_runtime()
    return return_value
