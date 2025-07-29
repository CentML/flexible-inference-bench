import argparse
import json
import logging
import random
import asyncio
import itertools
import sys
import os
import time
from contextlib import nullcontext
from typing import List, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import base64
import uuid
import io
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # type: ignore[attr-defined]
from flexible_inference_benchmark.engine.distributions import DISTRIBUTION_CLASSES, Distribution, Same, UniformInt
from flexible_inference_benchmark.utils.utils import (
    configure_logging,
    try_find_model,
    try_find_endpoint,
    set_max_open_files,
    download_sharegpt_dataset,
)
from flexible_inference_benchmark.utils.tokenizer import select_tokenizer
from flexible_inference_benchmark.utils.image_preprocesing import change_image_pixels
from flexible_inference_benchmark.engine.data import ShareGPT, Textfile, Random
from flexible_inference_benchmark.engine.client import Client
from flexible_inference_benchmark.engine.backend_functions import ASYNC_REQUEST_FUNCS
from flexible_inference_benchmark.engine.workloads import WORKLOADS_TYPES
from flexible_inference_benchmark.data_postprocessors.performance import add_performance_parser, calculate_metrics
from flexible_inference_benchmark.data_postprocessors.ttft import add_ttft_parser
from flexible_inference_benchmark.data_postprocessors.itl import add_itl_parser
from flexible_inference_benchmark.utils.telemetry import setup_telemetry
from opentelemetry import trace
from opentelemetry.trace import SpanKind

logger = logging.getLogger(__name__)

# Default value for num_trials argument
DEFAULT_NUM_TRIALS = 10
MAX_TRIALS = 100  # Maximum trials for prompt generation, warn if exceeded


def return_random_image_by_size(width: int, height: int, convert_to_base64: bool = False) -> Any:

    image_url = f"https://picsum.photos/{width}/{height}"
    if convert_to_base64:
        image_size = (width, height)
        channels = 3
        random_bytes = os.urandom(width * height * channels)
        random_image = Image.frombytes('RGB', image_size, random_bytes)
        buffered = io.BytesIO()
        random_image.save(buffered, format="JPEG")
        base64_encoded_image = base64.b64encode(buffered.getvalue())
        return base64_encoded_image
    else:
        return image_url


def parse_tuple(value: str) -> List[Tuple[int, int]]:
    """
    Parses a string of width-height pairs into a list of tuples of ints.

    Example:
        "1280x720,256x256" -> [(1280, 720),(256, 256)]
    """
    try:
        return [(int(width), int(height)) for part in value.split(',') for width, height in [part.split('x')]]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            (
                f"Invalid format: {value}. Must be a single string with "
                "'width1 x height1,...,widthN x heightN' pairs, e.g., '256x256,512x512'"
            )
        ) from e


# Specify the number and dimensions of images to be attached to each request
def generate_request_media(
    num_of_imgs_per_req: int,
    img_ratios_per_req: List[Tuple[int, int]],
    img_base_path: Union[str, None],
    size: int,
    send_image_with_base64: bool = False,
) -> List[List[List[str]]]:

    num_imgs_per_req = num_of_imgs_per_req
    if not num_imgs_per_req:
        return [[[] for _ in range(size)]]

    results: List[List[List[str]]] = []
    for ratios in img_ratios_per_req:
        media_per_request: List[List[str]] = []

        img_cntr = 0

        def _process_sample() -> None:
            media_file: Any = None
            nonlocal img_cntr
            # If img_base_path is provided, store the image locally
            # Otherwise, feed the image online
            if img_base_path:
                assert not send_image_with_base64, "Base64 encoding is not supported for local images"
                # If an image doesn't exist, download it
                img_path = os.path.join(img_base_path, f"{ratios[0]}x{ratios[1]}_{img_cntr + 1}.jpg")
                if not os.path.exists(img_path):
                    os.makedirs(img_base_path, exist_ok=True)
                    logger.info(f"Downloading image to {img_path} ...")
                    img_url = return_random_image_by_size(ratios[0], ratios[1])
                    img_data = requests.get(img_url, timeout=60).content
                    with open(img_path, 'wb') as handler:
                        handler.write(img_data)
                media_file = 'file://' + img_path
            else:
                # Fetch the image online with the ratios
                media_file = return_random_image_by_size(ratios[0], ratios[1], convert_to_base64=send_image_with_base64)

            img_cntr += 1
            if send_image_with_base64:
                media_arr = change_image_pixels(media_file, iterations=num_imgs_per_req)  # type: ignore[arg-type]
            else:
                media_arr = [media_file] * num_imgs_per_req
            media_per_request.append(media_arr)

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(_process_sample) for _ in range(size)]
            for future in tqdm(futures, desc=f"Generating images for {ratios[0]}x{ratios[1]}", total=size):
                future.result()

        results.append(media_per_request)
    return results


def select_distribution(args: List[Any]) -> Union[Distribution, Any]:
    dist_type = args[0]
    dist_args = (float(i) for i in args[1:])
    return DISTRIBUTION_CLASSES[dist_type](*dist_args)


def generate_request_times(args: argparse.Namespace) -> List[Union[int, float]]:
    if args.num_of_req:
        size = args.num_of_req
        dist = select_distribution(args.request_distribution)
        requests_times = dist.generate_distribution(size)
        return requests_times
    else:
        size = 1
        dist = select_distribution(args.request_distribution)
        # Check if any elements exceed max length
        while size < 1e6 and not [i for i in dist.generate_distribution(size) if i > args.max_time_for_reqs]:
            size *= 2
        requests_times = dist.generate_distribution(size)
        if size >= 1e6:
            size = int(1e6)
            # Eg. if the user runs `--timeout 10 -rps inf`, an arbitrary number of requests at time=0 can be generated
            logger.warning("Capping number of requests at 1000`000")
        return [i for i in requests_times if i <= args.max_time_for_reqs]


def generate_prompts(
    args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase, size: int
) -> List[Tuple[str, int, int]]:
    filename = args.dataset_path
    prompt_cls: Union[Random, Textfile, ShareGPT, None] = None

    input_prompt_dist = select_distribution(args.input_token_distribution) if args.input_token_distribution else None
    output_token_dist = select_distribution(args.output_token_distribution) if args.output_token_distribution else None
    if args.native_output_len:
        if output_token_dist is not None:
            raise ValueError(
                "Native output length is not compatible with output token distribution. "
                "The model will generate until EOS."
            )
        logger.info(
            "User selected native output length. "
            "Ignoring output token distribution and disabling ignore-eos. "
            "Output length will be capped to 8192 completion tokens."
        )
        args.disable_ignore_eos = True

        output_token_dist = Same(8192)

    if args.dataset_name.startswith('sharegpt'):
        if args.input_token_distribution is not None:
            raise ValueError(
                "Input token distribution is not supported with ShareGPT dataset. "
                "The prompts are already pre-defined in the dataset."
            )
        logger.info(
            "User selected sharegpt dataset. "
            "Ignoring prompt length distribution and following the prompts from the dataset."
        )
        if args.num_trials != DEFAULT_NUM_TRIALS:  # Check if user specified custom value
            logger.warning("num_trials parameter is ignored for ShareGPT dataset as prompts are pre-defined")
        prompt_cls = ShareGPT(filename, tokenizer, output_token_dist)
    else:
        logger.info(f"User selected {args.dataset_name} dataset. Generating prompt from distributions.")
        if input_prompt_dist is None:
            logger.info(
                "Input token distribution not provided. Defaulting to uniform distribution from 1 to 255 tokens."
            )
            input_prompt_dist = UniformInt(1, 255)

        if output_token_dist is None:
            logger.info(
                "Output token distribution not provided. Defaulting to uniform distribution from 1 to 255 tokens."
            )
            output_token_dist = UniformInt(1, 255)

        if args.prefix_len:
            prompt_cls = (
                Random.with_prefix_len(
                    args.prefix_len,
                    input_prompt_dist,
                    output_token_dist,
                    tokenizer,
                    args.ignore_input_distribution,
                    args.num_trials,
                )
                if args.dataset_name == "random"
                else Textfile.with_prefix_len(
                    filename,
                    args.prefix_len,
                    input_prompt_dist,
                    output_token_dist,
                    tokenizer,
                    args.ignore_input_distribution,
                    args.num_trials,
                )
            )
        else:
            prefix_text = args.prefix_text or ""
            prompt_cls = (
                Random.with_prefix_str(
                    prefix_text,
                    input_prompt_dist,
                    output_token_dist,
                    tokenizer,
                    args.ignore_input_distribution,
                    args.num_trials,
                )
                if args.dataset_name == "random"
                else Textfile.with_prefix_str(
                    filename,
                    prefix_text,
                    input_prompt_dist,
                    output_token_dist,
                    tokenizer,
                    args.ignore_input_distribution,
                    args.num_trials,
                )
            )

    if not prompt_cls:
        logger.error("Error generating prompts, exiting benchmark .....")
        sys.exit(1)

    factor = 1.2
    size_adjusted = int(size * factor)

    data = prompt_cls.generate_data(size_adjusted)
    if len(data) < size:
        logger.warning("The number of requests is less than the size.")
    else:
        data = data[:size]
    return data


def send_requests(
    client: Client,
    requests_prompts: List[Tuple[str, int, int]],
    requests_times: List[Union[int, float]],
    requests_media: List[List[str]],
) -> List[Any]:
    return asyncio.run(client.benchmark(requests_prompts, requests_times, requests_media))


def add_benchmark_subparser(subparsers: argparse._SubParsersAction) -> Any:  # type: ignore [type-arg]

    benchmark_parser = subparsers.add_parser(
        'benchmark', help="Benchmark an LLM serving endpoint", usage="fib benchmark [options]"
    )

    benchmark_parser.add_argument("--seed", type=int, default=None, help="seed for reproducibility")

    benchmark_parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default='openai',
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="Backend inference engine.",
    )

    benchmark_parser.add_argument(
        "-w",
        "--workload-type",
        "--workload",
        type=str,
        default=None,
        choices=list(WORKLOADS_TYPES.keys()),
        help="Preset request length distributions based on common workload types.",
    )

    url_group = benchmark_parser.add_mutually_exclusive_group()

    url_group.add_argument("--base-url", type=str, default=None, help="Server base URL.")

    benchmark_parser.add_argument(
        "--https-ssl", default=True, help="Whether to check SSL certificates for HTTPS endpoints, default is True"
    )

    benchmark_parser.add_argument("--endpoint", type=str, help="API endpoint.")

    req_group = benchmark_parser.add_mutually_exclusive_group()

    req_group.add_argument("-n", "--num-of-req", type=int, default=None, help="Total number of request.")

    req_group.add_argument(
        "--max-time-for-reqs", "--timeout", type=int, default=None, help="Max time for requests in seconds."
    )

    benchmark_parser.add_argument(
        "--num-of-imgs-per-req",
        type=int,
        default=None,
        help="Number of images to attach to each request. Example: '3'.",
    )

    benchmark_parser.add_argument(
        "--img-ratios-per-req",
        type=parse_tuple,
        default='500x500',
        help=(
            "Single string with image aspect ratios (width x height) separated by commas "
            "to attach per request. Example: '256x256,500x500'."
        ),
    )

    benchmark_parser.add_argument(
        "--img-base-path",
        type=str,
        default=None,
        help="Base image directory. Example: '/path/to/imgs/'. If provided, images will be downloaded to"
        " this directory before benchmarking and fed from here. If not provided, images will be fed online,"
        " which could cause excessive network delays in large numbers. To enable this, the serving engine"
        " also needs to start with the --allowed-local-media-path /path/to/imgs/ option.",
    )

    benchmark_parser.add_argument(
        "--num-validation-reqs",
        type=int,
        default=1,
        help="Number of requests to send for validation and warmup before the benchmark.",
    )

    benchmark_parser.add_argument(
        "--send-image-with-base64",
        action="store_true",
        help="Send images as base64 encoded strings. This is useful for testing the server's ability to handle"
        " base64 encoded images. If this is set, the --img-base-path option will be ignored.",
    )

    benchmark_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Optional limit on the number of concurrent in-flight requests.",
    )

    benchmark_parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Number of logprobs to return with each completion. Default is None, meaning no logprobs.",
    )

    benchmark_parser.add_argument(
        "--request-distribution",
        nargs="*",
        default=["exponential", 1],
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    benchmark_parser.add_argument(
        "-rps",
        "--requests-per-second",
        dest='request_distribution',
        type=lambda n: ["poisson", n],
        help="Presets the request distribution to N requests per second following a poisson distribution.",
    )

    benchmark_parser.add_argument(
        "--input-token-distribution",
        nargs="*",
        default=None,
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    benchmark_parser.add_argument(
        "--ignore-input-distribution",
        action="store_true",
        help="Ignore the input token distribution. This is meant to be used with --prefix-len or --prefix-text.",
    )

    benchmark_parser.add_argument(
        "--output-token-distribution",
        nargs="*",
        default=None,
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    benchmark_parser.add_argument(
        "--native-output-len",
        action="store_true",
        help="If set, the output token distribution will be ignored and the generation will continue until EOS."
        "This option is not compatible with output-token-distribution, and sets --disable-ignore-eos.",
    )

    benchmark_parser.add_argument(
        "--varying-requests",
        "--wave",
        type=int,
        nargs=3,
        dest="wave",
        help="Send requests at a varying request concurrency in a wave-like pattern",
    )

    prefix_group = benchmark_parser.add_mutually_exclusive_group()

    prefix_group.add_argument("--prefix-text", type=str, default=None, help="Text to use as prefix for all requests.")

    prefix_group.add_argument("--prefix-len", type=int, default=None, help="Length of prefix to use for all requests.")

    benchmark_parser.add_argument("--disable-ignore-eos", action="store_true", help="Disables ignoring the eos token.")

    benchmark_parser.add_argument("--disable-stream", action="store_true", help="Disable stream response from API.")

    benchmark_parser.add_argument("--cookies", default={}, help="Insert cookies in the request.")

    benchmark_parser.add_argument(
        "--dataset-name",
        "--dataset",
        type=str,
        default="random",
        choices=["sharegpt", "sharegpt_code", "other", "random"],
        help="Name of the dataset to benchmark on.",
    )

    benchmark_parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset.")

    benchmark_parser.add_argument("-m", "--model", type=str, help="Name of the model.")

    benchmark_parser.add_argument(
        "--tokenizer", type=str, default=None, help="Name or path of the tokenizer, if not using the default tokenizer."
    )

    benchmark_parser.add_argument(
        "--tokenizer-mode", type=str, default=None, help="Specify tokenizer mode. Eg. mistral. Default None"
    )

    benchmark_parser.add_argument("--disable-tqdm", action="store_true", help="Specify to disable tqdm progress bar.")

    benchmark_parser.add_argument("--best-of", type=int, default=1, help="Number of best completions to return.")

    benchmark_parser.add_argument("--use-beam-search", action="store_true", help="Use beam search for completions.")

    benchmark_parser.add_argument(
        "--json-response", action="store_true", help="Request responses in JSON format from the API."
    )

    benchmark_parser.add_argument(
        "--json-prompt",
        type=str,
        default="",
        help="Custom prompt message to append when using JSON modes. "
        "Supports inline text or file input with @file syntax (e.g., --json-prompt @prompt.txt). "
        "Always appended when specified, regardless of JSON mode type.",
    )

    benchmark_parser.add_argument(
        "--include-schema-in-prompt",
        action="store_true",
        help="Include the JSON schema in the prompt text for better LLM comprehension. "
        "Requires either --json-schema-file or --json-schema-inline to be specified.",
    )

    json_group = benchmark_parser.add_mutually_exclusive_group()
    json_group.add_argument(
        "--json-schema-file", type=str, help="Path to JSON schema file for structured output validation."
    )
    json_group.add_argument(
        "--json-schema-inline", type=str, help="Inline JSON schema string for structured output validation."
    )

    benchmark_parser.add_argument(
        "--disable-thinking", action="store_true", help="Disable thinking mode in chat templates."
    )

    benchmark_parser.add_argument(
        "--output-file",
        type=str,
        default='output-file.json',
        required=False,
        help="Output json file to save the results.",
    )

    benchmark_parser.add_argument("--debug", action="store_true", help="Log debug messages.")

    benchmark_parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with " "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )

    benchmark_parser.add_argument("--verbose", action="store_true", help="Print short description of each request.")

    benchmark_parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Temperature for sampling.")

    benchmark_parser.add_argument("--top-p", type=float, default=None, help="Top-p for sampling.")

    benchmark_parser.add_argument("--top-k", type=int, default=None, help="Top-k for sampling.")

    benchmark_parser.add_argument("-c", "--config-file", default=None, help="Configuration file.")

    benchmark_parser.add_argument(
        "--validation-prompt-tokens",
        type=int,
        default=128,
        help="Number of input tokens to use for validation prompts (default: 128).",
    )

    benchmark_parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Number of attempts to achieve exact token count when generating prompts (default: 10). "
        "Used for 'random' and 'other' datasets. Higher values improve token count precision "
        "but may slow down prompt generation. Ignored for ShareGPT datasets.",
    )

    return benchmark_parser


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="CentML Inference Benchmark")

    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand', required=True)

    add_performance_parser(subparsers)
    benchmark_parser = add_benchmark_subparser(subparsers)
    add_ttft_parser(subparsers)
    add_itl_parser(subparsers)

    args = parser.parse_args()
    if args.subcommand == 'benchmark':
        if args.config_file:
            with open(args.config_file, 'r') as f:
                file_data = json.load(f)
            for k, v in file_data.items():
                # Reload arguments to override config file values with command line values
                setattr(args, k, v)

        configure_logging(args)

        def fail(msg: str) -> None:
            benchmark_parser.print_help()
            print('\n\n\n')
            logger.error(msg)
            sys.exit(1)

        if args.wave:
            if not args.num_of_req:
                fail("Number of requests must be provided for varying requests")
            if args.wave[0] >= args.wave[1]:
                fail("Min wave concurrency must be smaller than max wave concurrency")
            if args.wave[0] <= 0:
                fail("Min wave concurrency must be positive")
            if args.wave[2] <= 0:
                fail("Wave sustain must be positive")
            if args.max_concurrent:
                logger.warning("Both varying requests and max concurrency provided. Ignoring max concurrency")
                args.max_concurrent = None
            if args.request_distribution:
                logger.warning(
                    "Both varying requests and request rate/distribution provided. Ignoring request rate/distribution"
                )
                args.request_distribution = None

            args.request_distribution = ["poisson", "inf"]

        if not (args.num_of_req or args.max_time_for_reqs):
            logger.info("Number of requests and max time for requests not provided. Defaulting to 1 request.")
            args.num_of_req = 1

        openapi = None
        if not args.base_url or not args.model or not args.endpoint:
            if not args.base_url:
                logger.info("Base url not provided. Searching for ports on localhost...")
                base_try_options = ["http://localhost:8000", "http://localhost:8080"]
            else:
                base_try_options = [args.base_url]
            for base_url, path in itertools.product(base_try_options, ["openapi.json", "health", "openai/health"]):
                try:
                    response = requests.get(f"{base_url}/{path}", timeout=1)
                    response.raise_for_status()
                    args.base_url = base_url
                    if "openapi" in path:
                        openapi = response.json()
                    break
                except (requests.HTTPError, requests.ConnectionError):
                    continue
            if not args.base_url:
                fail("No server found. Please provide the base url.")
            logger.info(f"Server found at {args.base_url}. Continuing.")
        if not args.model:
            logger.info("Model name not provided. Trying to query the model name from the server.")
            model = try_find_model(args.base_url, openapi)
            if model is None:
                fail("Model could not be deduced automatically. Please provide the model name.")
            else:
                logger.info(f"Model identified: {model}")
                args.model = model
        if not args.endpoint:
            args.endpoint = try_find_endpoint(args.base_url, openapi)
        if args.endpoint and args.endpoint[0] != '/':
            args.endpoint = "/" + args.endpoint

        if not args.dataset_path and args.dataset_name.startswith('sharegpt'):
            # download the sharegpt dataset and cache it in the home directory
            cache_dir = os.path.expanduser("~/.cache/flexible_inference_benchmark/")
            dataset_filename = args.dataset_name + ".json"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            sharegpt_path = os.path.join(cache_dir, dataset_filename)
            if not os.path.exists(sharegpt_path):
                logger.info(
                    "Downloading the sharegpt dataset to ~/.cache/flexible_inference_benchmark/%s ...", dataset_filename
                )
                download_sharegpt_dataset(args.dataset_name, sharegpt_path)
            args.dataset_path = sharegpt_path

        if args.dataset_name.startswith('sharegpt') and args.workload_type:
            fail(
                "ShareGPT dataset is selected. "
                "Prompt and output distributions will be ignored. "
                "Do not specify workload type with ShareGPT dataset."
            )

        if args.dataset_path and not args.dataset_name:
            args.dataset_name = "other"

        # Validate num_trials parameter
        if args.num_trials <= 0:
            fail("Number of trials must be positive")
        if args.num_trials > MAX_TRIALS:
            logger.warning(f"High num_trials value ({args.num_trials}) may slow down prompt generation")

    return args


def generate_fixed_validation_prompt(
    tokenizer: PreTrainedTokenizerBase, target_tokens: int = 128
) -> Tuple[str, int, int]:
    """
    Generate a fixed-size validation prompt for consistent warmup behavior.

    Args:
        tokenizer: The tokenizer to use for token counting
        target_tokens: Target number of input tokens (default: 128)

    Returns:
        Tuple of (prompt_text, input_tokens, output_tokens)
    """
    # Simple repeatable text that tokenizes predictably
    base_text = "The quick brown fox jumps over the lazy dog. "

    # Estimate tokens and repeat text to reach target
    sample_tokens = len(tokenizer.encode(base_text))
    repetitions = max(1, target_tokens // sample_tokens)

    validation_prompt = base_text * repetitions
    actual_tokens = len(tokenizer.encode(validation_prompt))

    # Fixed output length for validation (small but meaningful)
    output_tokens = 32

    return validation_prompt, actual_tokens, output_tokens


def run_main(args: argparse.Namespace) -> None:
    if args.workload_type:
        workload_type = WORKLOADS_TYPES[args.workload_type]()
        workload_type.overwrite_args(args)
    logger.info(f"Arguments: {args}")
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Set up telemetry
    telemetry_enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    setup_telemetry()
    run_id = str(uuid.uuid4())

    # Create a top-level span for the entire experiment that is not a parent span
    tracer = trace.get_tracer(__name__)
    otel_span = (
        tracer.start_span(
            "experiment",
            kind=SpanKind.INTERNAL,
            attributes={
                "fib.run.id": run_id,
                "fib.command": json.dumps(
                    {
                        "subcommand": args.subcommand,
                        "args": {k: v for k, v in vars(args).items() if k not in ['subcommand'] and v is not None},
                    }
                ),
            },
        )
        if telemetry_enabled
        else nullcontext()
    )

    with otel_span as span:
        requests_times = generate_request_times(args)
        size = len(requests_times)
        requests_media = generate_request_media(
            args.num_of_imgs_per_req, args.img_ratios_per_req, args.img_base_path, size, args.send_image_with_base64
        )
        tokenizer_id = args.tokenizer if args.tokenizer else args.model
        tokenizer: PreTrainedTokenizerBase = select_tokenizer(tokenizer_id, args.tokenizer_mode)
        requests_prompts = generate_prompts(args, tokenizer, size)
        min_length = min(len(requests_prompts), len(requests_times))
        requests_prompts = requests_prompts[:min_length]
        requests_times = requests_times[:min_length]
        requests_media = [arr_dims[:min_length] for arr_dims in requests_media]

        set_max_open_files(min_length + 256)

        base_url = args.base_url.strip("/")
        endpoint = args.endpoint.strip("/")
        args.api_url = f"{base_url}/{endpoint}"

        # Process JSON prompt with @file support
        custom_prompt = ""
        if args.json_prompt:
            if args.json_prompt.startswith("@"):
                # File-based prompt loading
                prompt_file_path = args.json_prompt[1:]  # Remove @ prefix
                try:
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        custom_prompt = f.read().strip()
                    if not custom_prompt:
                        logger.error(f"Prompt file '{prompt_file_path}' is empty")
                        sys.exit(1)
                    logger.info(f"Loaded custom prompt from {prompt_file_path}")
                except FileNotFoundError:
                    logger.error(f"Prompt file '{prompt_file_path}' does not exist")
                    sys.exit(1)
                except UnicodeDecodeError as e:
                    logger.error(f"Cannot read prompt file '{prompt_file_path}': {e}")
                    sys.exit(1)
                except Exception as e:
                    logger.error(f"Failed to load prompt file '{prompt_file_path}': {e}")
                    sys.exit(1)
            else:
                # Inline prompt
                custom_prompt = args.json_prompt
                if not custom_prompt:
                    logger.warning("JSON prompt is empty")

        # Process JSON schema if provided
        json_schema = None
        if args.json_schema_file:
            try:
                with open(args.json_schema_file, 'r') as f:
                    json_schema = json.load(f)
                # Basic validation that it's a valid JSON schema structure
                if not isinstance(json_schema, dict):
                    logger.error("JSON schema must be a JSON object")
                    sys.exit(1)
                logger.info(f"Loaded JSON schema from {args.json_schema_file}")
            except Exception as e:
                logger.error(f"Failed to load JSON schema file: {e}")
                sys.exit(1)
        elif args.json_schema_inline:
            try:
                json_schema = json.loads(args.json_schema_inline)
                # Basic validation that it's a valid JSON schema structure
                if not isinstance(json_schema, dict):
                    logger.error("JSON schema must be a JSON object")
                    sys.exit(1)
                logger.info("Loaded inline JSON schema")
            except Exception as e:
                logger.error(f"Failed to parse inline JSON schema: {e}")
                sys.exit(1)

        # Comprehensive input validation
        # 1. Check for contradictory flag combinations
        if json_schema and args.json_response:
            logger.error("Cannot use both --json-response and JSON schema options together")
            logger.error("Suggestion: Choose either --json-response or --json-schema-file/--json-schema-inline")
            sys.exit(2)

        if args.json_schema_file and args.json_schema_inline:
            logger.error("Cannot use --json-schema-file and --json-schema-inline together")
            logger.error("Suggestion: Choose either file-based or inline schema input")
            sys.exit(2)

        # 2. Check for schema-dependent flags without schema
        if hasattr(args, 'include_schema_in_prompt') and args.include_schema_in_prompt:
            if not json_schema:
                logger.error("--include-schema-in-prompt requires a JSON schema")
                logger.error("Suggestion: Add --json-schema-file <file> or --json-schema-inline <schema>")
                sys.exit(3)

        # 3. File size warnings (optional)
        if args.json_schema_file:
            try:
                file_size = os.path.getsize(args.json_schema_file)
                if file_size > 1024 * 1024:  # 1MB
                    logger.warning(f"Large schema file ({file_size / (1024*1024):.1f}MB) may impact performance")
            except OSError:
                pass  # File size check is optional

        if args.json_prompt and args.json_prompt.startswith("@"):
            prompt_file_path = args.json_prompt[1:]
            try:
                file_size = os.path.getsize(prompt_file_path)
                if file_size > 100 * 1024:  # 100KB
                    logger.warning(f"Large prompt file ({file_size / 1024:.1f}KB) may impact performance")
            except OSError:
                pass  # File size check is optional

        client = Client(
            args.backend,
            args.api_url,
            args.base_url,
            args.model,
            args.best_of,
            args.use_beam_search,
            True if args.verbose else args.disable_tqdm,
            args.https_ssl,
            not args.disable_ignore_eos,
            not args.disable_stream,
            args.cookies,
            args.verbose,
            args.max_concurrent,
            args.wave,
            args.logprobs,
            args.temperature,
            args.top_p,
            args.top_k,
            run_id=run_id,
            json_response=args.json_response,
            custom_prompt=custom_prompt,
            disable_thinking=args.disable_thinking,
            json_schema=json_schema,
            include_schema_in_prompt=getattr(args, 'include_schema_in_prompt', False),
        )
        # disable verbose output for validation of the endpoint. This is done to avoid confusion on terminal output.
        client_verbose_value = client.verbose
        client.verbose = False

        # Generate fixed-size validation prompt instead of using first request
        validation_prompt_data = generate_fixed_validation_prompt(tokenizer, args.validation_prompt_tokens)
        validation_media = requests_media[0][0] if requests_media[0] else []

        logger.info(
            (
                f"Sending {args.num_validation_reqs} request(s) for validation and warmup "
                f"(fixed size: {validation_prompt_data[1]} input tokens)."
            )
        )
        for _ in range(args.num_validation_reqs):
            validate_endpoint = asyncio.run(client.validate_url_endpoint(validation_prompt_data, validation_media))
            if not validate_endpoint.success:
                logger.info(f"{validate_endpoint.error}.\nExiting benchmark ....")
                sys.exit(1)

        if args.profile:
            logger.info("Starting the Torch profiler.")
            asyncio.run(client.start_torch_profiler())

        client.verbose = client_verbose_value
        logger.info("Beginning benchmark.")

        for idx, arr_dims in enumerate(requests_media):
            if args.num_of_imgs_per_req:
                logger.info(
                    (
                        f"Benchmarking with {args.num_of_imgs_per_req} images per request "
                        f"with ratio {args.img_ratios_per_req[idx]}"
                    )
                )
            t = time.perf_counter()
            output_list: List[Any] = send_requests(client, requests_prompts, requests_times, arr_dims)
            benchmark_time = time.perf_counter() - t
            # pylint: disable=line-too-long
            output = {
                "backend": args.backend,
                "time": benchmark_time,
                "outputs": [request_func_output.model_dump() for request_func_output in output_list],  # type: ignore
                "inputs": requests_prompts,
                "tokenizer": args.tokenizer if args.tokenizer else args.model,
                "stream": not args.disable_stream,
            }

            # Calculate performance metrics and add them as span attributes
            metrics = calculate_metrics(
                output["inputs"], output["outputs"], output["time"], tokenizer, output["stream"]
            )
            # Add metrics as a single JSON blob attribute
            if span:
                span.set_attribute("fib.metrics", json.dumps(metrics))

            if args.output_file:
                filename = args.output_file
                if args.num_of_imgs_per_req:
                    w, h = args.img_ratios_per_req[idx]
                    filename = f"ratio_{w}x{h}_{filename}"
                with open(filename, "w") as f:
                    f.write(json.dumps(output, indent=4))  # type: ignore
            if args.debug:
                logger.debug(f"{output_list}")

        client.verbose = False
        if args.profile:
            logger.info("Stopping the Torch profiler.")
            asyncio.run(client.stop_torch_profiler())


def main() -> None:
    args = parse_args()
    if args.subcommand == "analyse":
        from flexible_inference_benchmark.data_postprocessors.performance import run

        run(args)
    elif args.subcommand == "generate-ttft-plot":
        from flexible_inference_benchmark.data_postprocessors.ttft import run

        run(args)
    elif args.subcommand == "generate-itl-plot":
        from flexible_inference_benchmark.data_postprocessors.itl import run

        run(args)
    elif args.subcommand == "benchmark":
        run_main(args)
    else:
        raise ValueError(f"Invalid subcommand {args.subcommand}")


if __name__ == '__main__':
    main()
