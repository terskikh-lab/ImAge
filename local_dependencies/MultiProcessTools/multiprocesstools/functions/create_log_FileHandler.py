import logging

logger = logging.getLogger("multiprocessing_helpers")
logger.setLevel(logging.DEBUG)


def create_log_FileHandler(job_name: str, handler_name: str, log_file: str) -> None:
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.DEBUG)
    ## create a file handler ##
    file_handler = logging.FileHandler(log_file)
    ## create a logging format ##
    formatter = logging.Formatter(
        "%(asctime)s:%(filename)s:%(funcName)s:%(name)s:%(levelname)s:%(message)s"
    )
    file_handler.set_name(handler_name)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    handler_names = [handler.get_name() for handler in logger.handlers]
    if handler_name in handler_names:
        logger.warning(
            f"{handler_name} already exists in logger {job_name}, replacing..."
        )
        logger.removeHandler(handler_name)
    logger.addHandler(file_handler)
