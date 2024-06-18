import pandas as pd
import os
import logging
import time
from ..config import OVERWRITE

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def save_dataframe_to_csv(df: pd.DataFrame, path: str, filename: str, **kwargs) -> None:
    try:
        name = os.path.join(path, filename)
        if os.path.exists(name):
            if OVERWRITE==False:
                logger.warning(f"{name} already exists, appending time...")
                name = os.path.join(path, str(time.time()) + filename)
            else:
                logger.warning("Overwriting file...")
                os.remove(name)
        df.to_csv(name, **kwargs)
    except:
        logger.error(f"Could not save {filename} to {path}")
