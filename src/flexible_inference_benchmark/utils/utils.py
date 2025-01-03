from typing import Optional

import logging
import argparse
import resource

logger = logging.getLogger(__name__)


def configure_logging(args: argparse.Namespace) -> None:

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG if args.debug else logging.INFO,
    )


# adapted from: https://github.com/vllm-project/vllm/blob/e1a5c2f0a123835558b1b1c9895181161527c55e/vllm/utils.py#L1857
def set_max_open_files(n: Optional[int]) -> None:
    """Try to expand the system limit on the maximum number of open files."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if n is not None and n > hard:
        logger.warning(
            "Number of requests exceeds maximum ulimit for number of open files"
            "Consider increasing the limit with ulimit -n"
        )
    target_amount = n or 65535
    request_amount = max(soft, min(hard, target_amount))
    if request_amount == soft:
        return

    logger.debug("Setting max open files limit to %d of upper limit %d", request_amount, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (request_amount, hard))
    except ValueError as e:
        logger.warning(
            "Failed to increase ulimit from %s to %s."
            "This can lead to errors for large amounts of requests."
            "Consider increasing the limit with ulimit -n."
            "Error: %s",
            soft,
            request_amount,
            e,
        )
