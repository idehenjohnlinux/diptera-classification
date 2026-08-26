from pathlib import Path
import logging
import logging.config

from src.utils.config import CONFIG


def setup_logging() -> None:
    """
    Configure the project's logging system.
    """

    log_config = CONFIG.log()

    # Create log directory automatically
    log_file = Path(log_config["handlers"]["file"]["filename"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(log_config)


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger instance.
    """
    return logging.getLogger(name)
