import logging
import argparse

logger = logging.getLogger(__name__)


def configure_logging(args: argparse.Namespace) -> None:
    show_debug_logs = hasattr(args, "debug") and args.debug

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG if show_debug_logs else logging.INFO,
    )
