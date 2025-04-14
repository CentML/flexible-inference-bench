import argparse
import json
import logging
import random
import asyncio
import itertools
import sys
import os
import time
from typing import List, Any, Tuple, Union
import requests
import numpy as np
from transformers import AutoTokenizer
from flexible_inference_benchmark.engine.distributions import DISTRIBUTION_CLASSES, Distribution
from flexible_inference_benchmark.utils.utils import (
    configure_logging,
    try_find_model,
    try_find_endpoint,
    set_max_open_files,
    download_sharegpt_dataset,
)
from flexible_inference_benchmark.engine.data import ShareGPT, Textfile, Random
from flexible_inference_benchmark.engine.client import Client
from flexible_inference_benchmark.engine.backend_functions import ASYNC_REQUEST_FUNCS
from flexible_inference_benchmark.engine.workloads import WORKLOADS_TYPES
from flexible_inference_benchmark.data_postprocessors.performance import add_performance_parser, calculate_metrics
from flexible_inference_benchmark.data_postprocessors.ttft import add_ttft_parser
from flexible_inference_benchmark.data_postprocessors.itl import add_itl_parser

logger = logging.getLogger(__name__)


def return_random_image_URL_by_size(width, height):
    return f"https://picsum.photos/{width}/{height}"


def parse_tuple(value):
    """
    Parses a width-height pair into a tuple of ints.
    
    Example:
        "1280x720" -> (1280, 720)
    """
    try:
        return tuple(map(int, value.split("x")))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid format: {value}. Must be a 'widthxheight' pair, e.g., '1920x1080' OR '1280x720'."
        )


# Specify the number and dimensions of images to be attached to each request
def generate_request_media(args: argparse.Namespace, size) \
    -> Union[List[List[Union[Tuple[int, int], str]]], None]:

    num_imgs_per_req = args.num_of_imgs_per_req
    if not num_imgs_per_req:
        return [[] for _ in range(size)]
    ratios = args.img_ratios_per_req

    media_per_request = []
    img_cntr = 0
    for i in range(size):
        media_per_request.append([])
        for j in range(int(num_imgs_per_req)):
            # If img_base_path is provided, store the image locally
            # Otherwise, feed the image online
            if args.img_base_path:
                # If an image doesn't exist, download it
                img_path = os.path.join(args.img_base_path, f"{ratios[0]}x{ratios[1]}_{img_cntr + 1}.jpg")
                if not os.path.exists(img_path):
                    os.makedirs(args.img_base_path, exist_ok=True)
                    logger.info(f"Downloading image to {img_path} ...")
                    img_url = return_random_image_URL_by_size(ratios[0], ratios[1])
                    img_data = requests.get(img_url).content
                    with open(img_path, 'wb') as handler:
                        handler.write(img_data)
                media_per_request[-1].append('file://' + img_path)
            else:
                # Fetch the image online with the ratios
                media_per_request[-1].append(return_random_image_URL_by_size(ratios[0], ratios[1]))
            img_cntr += 1
    
    return media_per_request


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
        while not [i for i in dist.generate_distribution(size) if i > args.max_time_for_reqs]:
            size *= 2
        requests_times = dist.generate_distribution(size)
        return [i for i in requests_times if i <= args.max_time_for_reqs]


def generate_prompts(args: argparse.Namespace, tokenizer: AutoTokenizer, size: int) -> List[Tuple[str, int, int]]:
    filename = args.dataset_path
    prompt_cls: Union[Random, Textfile, ShareGPT, None] = None
    if args.dataset_name.startswith('sharegpt'):
        logger.info(
            "User selected sharegpt dataset. "
            "Ignoring prompt and output length distribution and following the shapes from the dataset."
        )
        prompt_cls = ShareGPT(filename, tokenizer)
    else:
        logger.info(
            f"User selected {args.dataset_name} dataset. Generating prompt and output lengths from distributions."
        )
        input_prompt_dist = select_distribution(args.input_token_distribution)
        output_token_dist = select_distribution(args.output_token_distribution)

        if args.prefix_len:
            prompt_cls = (
                Random.with_prefix_len(args.prefix_len, input_prompt_dist, output_token_dist, tokenizer)
                if args.dataset_name == "random"
                else Textfile.with_prefix_len(
                    filename, args.prefix_len, input_prompt_dist, output_token_dist, tokenizer
                )
            )
        else:
            prefix_text = args.prefix_text or ""
            prompt_cls = (
                Random.with_prefix_str(prefix_text, input_prompt_dist, output_token_dist, tokenizer)
                if args.dataset_name == "random"
                else Textfile.with_prefix_str(filename, prefix_text, input_prompt_dist, output_token_dist, tokenizer)
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
    requests_media: List[List[Union[Tuple[int, int], str]]],
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
        help="Number of images to attach to each request. Example: '3'."
    )

    benchmark_parser.add_argument(
        "--img-ratios-per-req",
        type=parse_tuple,
        default='500x500',
        help="Image aspect ratios (width x height) to attach per request. Example: '500x500'."
    )

    benchmark_parser.add_argument(
        "--img-base-path",
        type=str,
        default=None,
        help="Base image directory. Example: '/path/to/imgs/'. If provided, images will be downloaded to" \
        " this directory before benchmarking and fed from here. If not provided, images will be fed online," \
        " which could cause excessive network delays in large numbers. To enable this, the serving engine" \
        " also needs to start with the --allowed-local-media-path /path/to/imgs/ option.",
    )

    benchmark_parser.add_argument(
        "--num-validation-reqs",
        type=int,
        default=None,
        help="Number of requests to send for validation and warmup before the benchmark.",
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
        default=["uniform", 1, 255],
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    benchmark_parser.add_argument(
        "--output-token-distribution",
        nargs="*",
        default=["uniform", 1, 255],
        help="Request distribution [Distribution_type (inputs to distribution)]",
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

    benchmark_parser.add_argument("--disable-tqdm", action="store_true", help="Specify to disable tqdm progress bar.")

    benchmark_parser.add_argument("--best-of", type=int, default=1, help="Number of best completions to return.")

    benchmark_parser.add_argument("--use-beam-search", action="store_true", help="Use beam search for completions.")

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
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )

    benchmark_parser.add_argument("--verbose", action="store_true", help="Print short description of each request.")

    benchmark_parser.add_argument("-c", "--config-file", default=None, help="Configuration file.")

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

    return args


def run_main(args: argparse.Namespace) -> None:
    if args.workload_type:
        workload_type = WORKLOADS_TYPES[args.workload_type]()
        workload_type.overwrite_args(args)
    logger.info(f"Arguments: {args}")
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    requests_times = generate_request_times(args)
    size = len(requests_times)
    requests_media = generate_request_media(args, size)
    tokenizer_id = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    requests_prompts = generate_prompts(args, tokenizer, size)
    min_length = min(len(requests_prompts), len(requests_times))
    requests_prompts = requests_prompts[:min_length]
    requests_times = requests_times[:min_length]
    requests_media = requests_media[:min_length]

    set_max_open_files(min_length + 256)

    base_url = args.base_url.strip("/")
    endpoint = args.endpoint.strip("/")
    args.api_url = f"{base_url}/{endpoint}"

    client = Client(
        args.backend,
        args.api_url,
        base_url,
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
        args.logprobs,
    )
    # disable verbose output for validation of the endpoint. This is done to avoid confusion on terminal output.
    client_verbose_value = client.verbose
    client.verbose = False

    if args.profile:
        logger.info("Starting the Torch profiler.")
        asyncio.run(client.start_torch_profiler(requests_prompts[0], requests_media[0]))

    logger.info(f"Sending {args.num_validation_reqs} requests for validation and warmup.")
    for i in range(args.num_validation_reqs):
        validate_endpoint = asyncio.run(client.validate_url_endpoint(requests_prompts[0], requests_media[0]))
        if not validate_endpoint.success:
            logger.info(f"{validate_endpoint.error}.\nExiting benchmark ....")
            sys.exit()
    client.verbose = client_verbose_value
    logger.info("Beginning benchmark.")
    t = time.perf_counter()
    output_list: List[Any] = send_requests(client, requests_prompts, requests_times, requests_media)
    benchmark_time = time.perf_counter() - t

    if args.profile:
        logger.info("Stopping the Torch profiler.")
        asyncio.run(client.stop_torch_profiler(requests_prompts[0], requests_media[0]))

    # pylint: disable=line-too-long
    output = {
        "backend": args.backend,
        "time": benchmark_time,
        "outputs": [request_func_output.model_dump() for request_func_output in output_list],  # type: ignore
        "inputs": requests_prompts,
        "tokenizer": args.tokenizer if args.tokenizer else args.model,
        "stream": not args.disable_stream,
    }

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(json.dumps(output, indent=4))  # type: ignore
    if args.debug:
        logger.debug(f"{output_list}")

    calculate_metrics(output["inputs"], output["outputs"], output["time"], tokenizer, output["stream"])


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
