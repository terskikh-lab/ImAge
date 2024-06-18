import logging

logger = logging.getLogger("multiprocessing_helpers")
logger.setLevel(logging.DEBUG)


def close_logger(job_name):
    logger = logging.getLogger(job_name)
    logger.info("Closing loggers")
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
