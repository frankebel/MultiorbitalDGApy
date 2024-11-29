import logging
from functools import wraps
from datetime import datetime

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
