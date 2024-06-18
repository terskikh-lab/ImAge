import os
import logging

from .create_log_FileHandler import create_log_FileHandler
from .create_log_StreamHandler import create_log_StreamHandler

logger = logging.getLogger("multiprocessing_helpers")
logger.setLevel(logging.DEBUG)


def create_logger_handlers(
    job_name: str, log_name: str, log_path: str, instance_number: int
) -> None:
    logger = logging.getLogger(job_name)
    log_file = os.path.join(log_path, log_name)
    create_log_FileHandler(job_name, f"file_handler{instance_number}", log_file)
    create_log_StreamHandler(job_name, f"stream_handler{instance_number}")
    logger.info(f"Created log file: {log_file}")
