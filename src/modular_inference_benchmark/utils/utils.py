import logging
import argparse

logger = logging.getLogger(__name__)


def configure_logging(args: argparse.Namespace) -> None:
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.DEBUG if args.debug else logging.INFO,
    }

    logging.basicConfig(**kwargs)  # type: ignore
