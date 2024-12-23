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

def try_find_model(base_url: str, openapi: Optional[dict]) -> Optional[str]:
    """
    Query the base URL for models and return the first model ID. Assumes an OpenAI-like API.

    Args:
        base_url (str): The URL to query.
        openapi (dict): Optional JSON object describing the API.

    Returns:
        Optional[str]: The first model ID, or None if the request fails.
    """
    try_paths = ["/v1/models", "/openai/v1/models", "/models"]
    if openapi:
        try_paths = filter(lambda path: path in openapi["paths"].keys(), try_paths)
    try_urls = [f"{base_url}{path}" for path in try_paths]
    resp_json = None
    for url in try_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            resp_json = response.json()
            break
        except:
            continue

    if resp_json is not None and "models" in resp_json:
        return resp_json["models"][0]["id"]
    elif resp_json is not None and "data" in resp_json:
        return resp_json["data"][0]["id"]
    return None

def try_find_endpoint(base_url: str, openapi: Optional[dict]) -> Optional[str]:
    """
    Query the base URL for endpoints and return any that exist.

    Args:
        base_url (str): The URL to query.
        openapi (dict): Optional JSON object describing the API.

    Returns:
        Optional[str]: The path of the identified endpoint, or None if no endpoint was found.
    """
    try_paths = ["/v1/completions", "/openai/v1/completions", "/generate_stream", "/v1/generate", "/generate"]
    if openapi:
        try_paths = filter(lambda path: path in openapi["paths"].keys(), try_paths)
    for path in try_paths:
        try:
            response = requests.post(f"{base_url}{path}")
            if response.ok or response.status_code == 400:
                return path # 400 means the endpoint exists, just that the request wasn't proper
            response.raise_for_status()
            break
        except:
            continue