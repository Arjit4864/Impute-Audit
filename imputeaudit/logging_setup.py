"""Rotating file logging shared by the interface and the runner."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from imputeaudit import config


def get_logger(name: str = "imputeaudit") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(config.LOG_DIR / "imputeaudit.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
