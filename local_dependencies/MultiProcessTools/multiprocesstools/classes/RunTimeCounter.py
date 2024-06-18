import time
import logging

logger = logging.getLogger("MultiProcessTools")
logger.setLevel(logging.DEBUG)


class RunTimeCounter:
    """
    Creates a runtime counter, which has the ability to check if the runtime has exceeded the timeout.
    """

    def __init__(self, timeout: float) -> None:
        self.start_queue = time.perf_counter()
        self.timeout = timeout

    def get_time_left(self) -> float:
        return time.perf_counter() - self.start_queue > self.timeout

    def check_runtime(self) -> None:
        if time.perf_counter() - self.start_queue > self.timeout:
            logger.error(f"Timeout reached. Illumination correction model not found")
            raise RuntimeError(
                f"Timeout exceeded while waiting for illumination correction model to be created"
            )
        else:
            return self.get_time_left()

    def __str__(self) -> str:
        return f"RunTimeCounter created at {self.start_queue} with timeout {self.timeout} and {self.get_time_left()} seconds left"
