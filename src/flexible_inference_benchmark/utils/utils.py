from typing import Any, Optional

import logging
import argparse
import requests
import resource

logger = logging.getLogger(__name__)


def configure_logging(args: argparse.Namespace) -> None:
    show_debug_logs = hasattr(args, "debug") and args.debug

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG if show_debug_logs else logging.INFO,
    )


def try_find_model(base_url: str, openapi: Optional[Any]) -> Optional[str]:
    """
    Query the base URL for models and return the first model ID. Assumes an OpenAI-like API.

    Args:
        base_url (str): The URL to query.
        openapi (Any): Optional JSON object describing the API.

    Returns:
        Optional[str]: The first model ID, or None if the request fails.
    """
    try_paths = ["/v1/models", "/openai/v1/models", "/models"]
    if openapi:
        try_paths = list(filter(lambda path: path in openapi["paths"].keys(), try_paths))
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
        return str(resp_json["models"][0]["id"])
    elif resp_json is not None and "data" in resp_json:
        return str(resp_json["data"][0]["id"])
    return None


def try_find_endpoint(base_url: str, openapi: Optional[Any]) -> Optional[str]:
    """
    Query the base URL for endpoints and return any that exist.

    Args:
        base_url (str): The URL to query.
        openapi (Any): Optional JSON object describing the API.

    Returns:
        Optional[str]: The path of the identified endpoint, or None if no endpoint was found.
    """
    try_paths = ["/v1/completions", "/openai/v1/completions", "/generate_stream", "/v1/generate", "/generate"]
    if openapi:
        try_paths = list(filter(lambda path: path in openapi["paths"].keys(), try_paths))

    if len(try_paths) == 1:
        return try_paths[0]

    for path in try_paths:
        try:
            response = requests.post(f"{base_url}{path}")
            if response.ok or response.status_code == 400:
                return path  # 400 means the endpoint exists, just that the request wasn't proper
            response.raise_for_status()
            break
        except:
            continue

    return None


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

def download_sharegpt_dataset(path: str) -> None:
    """
    Download the ShareGPT V3 dataset.

    Args:
        path (str): The path to save the dataset to.
    """
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)