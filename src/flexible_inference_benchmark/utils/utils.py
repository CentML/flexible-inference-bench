from typing import Optional

import logging
import argparse
import requests

logger = logging.getLogger(__name__)


def configure_logging(args: argparse.Namespace) -> None:
    show_debug_logs = hasattr(args, "debug") and args.debug

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG if show_debug_logs else logging.INFO,
    )

def try_find_model(base_url) -> Optional[str]:
    """
    Query the base URL for models and return the first model ID. Assumes an OpenAI-like API.

    Args:
        base_url (str): The URL to query.

    Returns:
        Optional[str]: The first model ID, or None if the request fails.
    """

    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        return response.json()["data"][0]["id"]
    except Exception as e:
        return None