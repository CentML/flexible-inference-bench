from typing import Any
import logging

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    kwargs = {"format": "%(asctime)s %(levelname)-8s %(message)s", "datefmt": "%Y-%m-%d %H:%M", "level": logging.INFO}

    logging.basicConfig(**kwargs)
