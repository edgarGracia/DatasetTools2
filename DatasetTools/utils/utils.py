import functools
import logging
from enum import Enum, unique


@unique
class ColorSource(Enum):
    SOLID = "SOLID"
    LABEL = "LABEL"
    INSTANCE = "INSTANCE"
    KEYPOINTS = "KEYPOINTS"


@unique
class RelativePosition(Enum):
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


class _LoggingFormatter(logging.Formatter):
    """Logging colored formatter.
    """
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format = "%Y-%d-%m %H:%M:%S"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.date_format)
        return formatter.format(record)


@functools.lru_cache()
def get_logger(
    name: str = "DatasetTools",
    module_name: str = "",
) -> logging.Logger:
    name = f"{name}.{module_name}" if module_name else name
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setFormatter(_LoggingFormatter())
    # if (logger.hasHandlers()):
    #     logger.handlers.clear()
    logger.addHandler(ch)
    logger.propagate = False
    return logger
