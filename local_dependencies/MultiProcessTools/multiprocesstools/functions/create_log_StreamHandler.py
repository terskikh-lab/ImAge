import logging

logger = logging.getLogger("multiprocessing_helpers")
logger.setLevel(logging.DEBUG)


def create_log_StreamHandler(job_name: str, handler_name: str) -> None:
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.DEBUG)
    ## create a file handler ##
    stream_handler = logging.StreamHandler()
    ## create a logging format ##
    formatter = logging.Formatter(
        "%(asctime)s:%(filename)s:%(funcName)s:%(name)s:%(levelname)s:%(message)s"
    )
    stream_handler.set_name(handler_name)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    handler_names = [handler.get_name() for handler in logger.handlers]
    if handler_name not in handler_names:
        logger.warning(f"{handler_name} already exists in logger {job_name}")
        logger.removeHandler(handler_name)
    logger.addHandler(stream_handler)
