"""Rotating logger setup shared by every module in the pipeline."""

import logging
from logging.handlers import RotatingFileHandler

from utils.paths import PIPELINE_LOG_PATH, ensure_directories

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        ensure_directories()
        root = logging.getLogger("chilli_platform")
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

        file_handler = RotatingFileHandler(
            PIPELINE_LOG_PATH, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        _CONFIGURED = True

    return logging.getLogger(f"chilli_platform.{name}")
