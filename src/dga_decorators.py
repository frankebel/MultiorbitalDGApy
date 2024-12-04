import logging
import time
from datetime import datetime
from functools import wraps

logger = logging.getLogger()


def log_debug(func):
    if not logger.hasHandlers():  # Avoid duplicate handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    @wraps(func)
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.debug(f"{timestamp} - Calling method: {func.__name__}.")
        result = func(*args, **kwargs)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.debug(f"{timestamp} - Method {func.__name__} completed.")
        return result

    return wrapper


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {(end_time - start_time):.6f} seconds")
        return result

    return wrapper
