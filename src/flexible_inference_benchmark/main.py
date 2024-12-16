import argparse
import json
import logging
import random
import asyncio
import sys
import time
from typing import List, Any, Tuple, Union
import numpy as np
from transformers import AutoTokenizer
from flexible_inference_benchmark.engine.distributions import DISTRIBUTION_CLASSES, Distribution
from flexible_inference_benchmark.utils.utils import configure_logging, try_find_model
from flexible_inference_benchmark.engine.data import ShareGPT, Textfile, Random
from flexible_inference_benchmark.engine.client import Client
from flexible_inference_benchmark.engine.backend_functions import ASYNC_REQUEST_FUNCS
from flexible_inference_benchmark.engine.workloads import WORKLOADS_TYPES
from flexible_inference_benchmark.data_postprocessors.performance import add_performance_parser
from flexible_inference_benchmark.data_postprocessors.ttft import add_ttft_parser
from flexible_inference_benchmark.data_postprocessors.itl import add_itl_parser

logger = logging.getLogger(__name__)


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


def generate_prompts(args: argparse.Namespace, size: int) -> List[Tuple[str, int, int]]:
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    filename = args.dataset_path
    prompt_cls: Union[Random, Textfile, ShareGPT, None] = None
    if args.dataset_name == 'sharegpt':
        logger.info(
            "User selected sharegpt dataset.\n \
            Ignoring prompt and output length distribution and following the shapes from the dataset.\n"
        )
        prompt_cls = ShareGPT(filename, tokenizer)
    else:
        logger.info(
            f"User selected {args.dataset_name} dataset. Generating prompt and output lengths from distributions"
        )
        input_prompt_dist = select_distribution(args.input_token_distribution)
        output_token_dist = select_distribution(args.output_token_distribution)

        if args.prefix_len or args.no_prefix:
            if args.prefix_len:
                prefix_len = args.prefix_len
            else:
                prefix_len = 0
            prompt_cls = (
                Random.with_prefix_len(prefix_len, input_prompt_dist, output_token_dist, tokenizer)
                if args.dataset_name == "random"
                else Textfile.with_prefix_len(filename, prefix_len, input_prompt_dist, output_token_dist, tokenizer)
            )
        else:
            prompt_cls = (
                Random.with_prefix_str(args.prefix_text, input_prompt_dist, output_token_dist, tokenizer)
                if args.dataset_name == "random"
                else Textfile.with_prefix_str(
                    filename, args.prefix_text, input_prompt_dist, output_token_dist, tokenizer
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
    client: Client, requests_prompts: List[Tuple[str, int, int]], requests_times: List[Union[int, float]]
) -> List[Any]:
    return asyncio.run(client.benchmark(requests_prompts, requests_times))


def add_benchmark_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore [type-arg]

    benchmark_parser = subparsers.add_parser('benchmark')

    benchmark_parser.add_argument("--seed", type=int, default=None, help="seed for reproducibility")

    benchmark_parser.add_argument(
        "--backend",
        type=str,
        default='cserve',
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="Backend inference engine.",
    )

    benchmark_parser.add_argument(
        "--workload-type",
        type=str,
        default=None,
        choices=list(WORKLOADS_TYPES.keys()),
        help="choose a workload type, this will overwrite some arguments",
    )

    url_group = benchmark_parser.add_mutually_exclusive_group()

    url_group.add_argument(
        "--base-url", type=str, default=None, help="Server or API base url if not using http host and port."
    )

    benchmark_parser.add_argument(
        "--https-ssl", default=True, help="whether to check for ssl certificate for https endpoints, default is True"
    )

    benchmark_parser.add_argument("--endpoint", type=str, default="/v1/completions", help="API endpoint.")

    req_group = benchmark_parser.add_mutually_exclusive_group()

    req_group.add_argument("--num-of-req", type=int, default=None, help="Total number of request.")

    req_group.add_argument("--max-time-for-reqs", type=int, default=None, help="Max time for requests in seconds.")

    benchmark_parser.add_argument(
        "--request-distribution",
        nargs="*",
        default=["exponential", 1],
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    benchmark_parser.add_argument(
        "--input-token-distribution",
        nargs="*",
        default=["uniform", 0, 255],
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    benchmark_parser.add_argument(
        "--output-token-distribution",
        nargs="*",
        default=["uniform", 0, 255],
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    prefix_group = benchmark_parser.add_mutually_exclusive_group()

    prefix_group.add_argument("--prefix-text", type=str, default=None, help="Text to use as prefix for all requests.")

    prefix_group.add_argument("--prefix-len", type=int, default=None, help="Length of prefix to use for all requests.")

    prefix_group.add_argument('--no-prefix', type=bool, default=True, help='No prefix for requests, default is True.')

    benchmark_parser.add_argument("--disable-ignore-eos", action="store_true", help="Disables ignoring the eos token")

    benchmark_parser.add_argument("--disable-stream", action="store_true", help="Disable stream response from API")

    benchmark_parser.add_argument("--cookies", default={}, help="Insert cookies in the request")

    benchmark_parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["sharegpt", "other", "random"],
        help="Name of the dataset to benchmark on.",
    )

    benchmark_parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset.")

    benchmark_parser.add_argument("--model", type=str, help="Name of the model.")

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

    benchmark_parser.add_argument("--debug", action="store_true", help="Log debug messages")

    benchmark_parser.add_argument("--verbose", action="store_true", help="Print short description of each request")

    benchmark_parser.add_argument("--config-file", default=None, help="configuration file")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="CentML Inference Benchmark")

    subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand', required=True)

    add_performance_parser(subparsers)
    add_benchmark_subparser(subparsers)
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
        if not (args.prefix_text or args.prefix_len or args.no_prefix):
            parser.error("Please provide either prefix text or prefix length or specify no prefix.")
        if not (args.num_of_req or args.max_time_for_reqs):
            logger.info("Number of requests and max time for requests not provided. Defaulting to 1 request.")
            args.num_of_req = 1
        if not args.base_url:
            logger.info("Base url not provided. Defaulting to http://localhost:8000")
            args.base_url = "http://localhost:8000"
        if not args.model:
            logger.info("Model name not provided. Trying to query the model name from the server.")
            model = try_find_model(args.base_url)
            if model is None:
                parser.error("Model could not be deduced automatically. Please provide the model name.")
            else:
                logger.info(f"Model identified: {model}")
                args.model = model

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
    requests_prompts = generate_prompts(args, size)
    min_length = min(len(requests_prompts), len(requests_times))
    requests_prompts = requests_prompts[:min_length]
    requests_times = requests_times[:min_length]

    args.api_url = f"{args.base_url}{args.endpoint}"

    client = Client(
        args.backend,
        args.api_url,
        args.model,
        args.best_of,
        args.use_beam_search,
        True if args.verbose else args.disable_tqdm,
        args.https_ssl,
        not args.disable_ignore_eos,
        not args.disable_stream,
        args.cookies,
        args.verbose,
    )
    # disable verbose output for validation of the endpoint. This is done to avoid confusion on terminal output.
    client_verbose_value = client.verbose
    client.verbose = False
    validate_endpoint = asyncio.run(client.validate_url_endpoint(requests_prompts[0]))
    if not validate_endpoint.success:
        logger.info(f"{validate_endpoint.error}.\nExiting benchmark ....")
        sys.exit()
    client.verbose = client_verbose_value
    t = time.perf_counter()
    output_list: List[Any] = send_requests(client, requests_prompts, requests_times)
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

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(json.dumps(output, indent=4))  # type: ignore
    if args.debug:
        logger.debug(f"{output_list}")


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
