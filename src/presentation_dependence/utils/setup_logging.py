import logging
import sys
from datetime import datetime


def setup_logging(name: str, config=None, level=None, output_file=None) -> logging.Logger:
    """Return a configured stdlib logger.

    Args:
        name: logger name.
        config: optional config dict carrying a `logging:` sub-block with keys
            `level`, `disabled`, `log_file`.
        level: override for the logging level; ignored if logger is disabled.
        output_file: write a timestamped copy of the log stream here in
            addition to stdout. Takes precedence over `config["logging"]["log_file"]`.
    """
    config = config or {}
    logger = logging.getLogger(name)

    if getattr(logger, "is_configured", False):
        return logger

    if logger.hasHandlers():
        for handler in logger.handlers:
            logger.removeHandler(handler)

    logging_config = config.get("logging", {})
    logger.disabled = logging_config.get("disabled", False)
    if logger.disabled:
        return logger

    if level is None:
        level = logging._nameToLevel.get(logging_config.get("level", "INFO"), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(level)
    logger.addHandler(ch)

    if output_file is None:
        output_file = logging_config.get("log_file")

    if output_file:
        if output_file.endswith(".log"):
            output_file = output_file[:-4]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_file}_{timestamp}.log"
        oh = logging.FileHandler(output_file)
        oh.setFormatter(formatter)
        oh.setLevel(level)
        logger.addHandler(oh)

    logger.propagate = False
    logger.is_configured = True
    return logger
