import logging
from pathlib import Path


def setup_logger(name, log_file):
    # creating logs folder
    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # log format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # file handler
        file_handler = logging.FileHandler(logs_dir / log_file)
        file_handler.setFormatter(formatter)

        # console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # add log handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
