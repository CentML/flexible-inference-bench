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
from modular_inference_benchmark.engine.distributions import DISTRIBUTION_CLASSES, Distribution
from modular_inference_benchmark.utils.utils import configure_logging
from modular_inference_benchmark.engine.data import ShareGPT, Textfile, Random
from modular_inference_benchmark.engine.client import Client
from modular_inference_benchmark.engine.backend_functions import ASYNC_REQUEST_FUNCS

logger = logging.getLogger(__name__)


def select_distribution(args: List[Any]) -> Union[Distribution, Any]:
    dist_type = args[0]
    dist_args = (float(i) for i in args[1:])
    return DISTRIBUTION_CLASSES[dist_type](*dist_args)


def generate_request_times(args: argparse.Namespace) -> List[int | float]:
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


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="CentML Inference Benchmark")

    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")

    parser.add_argument(
        "--backend",
        type=str,
        default='cserve',
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="Backend inference engine.",
    )

    url_group = parser.add_mutually_exclusive_group()

    url_group.add_argument(
        "--base-url", type=str, default=None, help="Server or API base url if not using http host and port."
    )

    url_group.add_argument(
        "--host-port", type=str, default="localhost:8080", help="Host and port for the server in host:port format"
    )

    parser.add_argument("--endpoint", type=str, default="/v1/completions", help="API endpoint.")

    req_group = parser.add_mutually_exclusive_group()

    req_group.add_argument("--num-of-req", type=int, default=None, help="Total number of request.")

    req_group.add_argument("--max-time-for-reqs", type=int, default=None, help="Max time for requests in seconds.")

    parser.add_argument(
        "--request-distribution",
        nargs="*",
        default=["exponential", 1],
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    parser.add_argument(
        "--input-token-distribution",
        nargs="*",
        default=["uniform", 0, 255],
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    parser.add_argument(
        "--output-token-distribution",
        nargs="*",
        default=["uniform", 0, 255],
        help="Request distribution [Distribution_type (inputs to distribution)]",
    )

    prefix_group = parser.add_mutually_exclusive_group()

    prefix_group.add_argument("--prefix-text", type=str, default=None, help="Text to use as prefix for all requests.")

    prefix_group.add_argument("--prefix-len", type=int, default=None, help="Length of prefix to use for all requests.")

    prefix_group.add_argument('--no-prefix', action='store_true', help='No prefix for requests.')

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["sharegpt", "other", "random"],
        help="Name of the dataset to benchmark on.",
    )

    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset.")

    parser.add_argument("--model", type=str, help="Name of the model.")

    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Name or path of the tokenizer, if not using the default tokenizer."
    )

    parser.add_argument("--disable-tqdm", action="store_true", help="Specify to disable tqdm progress bar.")

    parser.add_argument("--best-of", type=int, default=1, help="Number of best completions to return.")

    parser.add_argument("--use-beam-search", action="store_true", help="Use beam search for completions.")

    parser.add_argument("--output-file", type=str, default=None, help="Output json file to save the results.")

    parser.add_argument("--debug", action="store_true", help="Log debug messages")

    parser.add_argument("--config-file", default=None, help="configuration file")
    args = parser.parse_args()
    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**json.load(f))
    args = parser.parse_args()
    if not (args.prefix_text or args.prefix_len or args.no_prefix):
        parser.error("Please provide either prefix text or prefix length or specify no prefix.")
    if not (args.num_of_req or args.max_time_for_reqs):
        parser.error("Please provide either number of requests or max time for requests.")
    if not args.model:
        parser.error("Please provide the model name.")
    return args


def main() -> None:
    args = parse_args()
    configure_logging(args)
    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    requests_times = generate_request_times(args)
    size = len(requests_times)
    requests_prompts = generate_prompts(args, size)
    min_length = min(len(requests_prompts), len(requests_times))
    requests_prompts = requests_prompts[:min_length]
    requests_times = requests_times[:min_length]

    if args.base_url is None:
        assert args.host_port, "Host and port must be provided if base url is not provided."
        args.api_url = f"http://{args.host_port}{args.endpoint}"
    else:
        args.api_url = f"{args.base_url}{args.endpoint}"

    client = Client(args.backend, args.api_url, args.model, args.best_of, args.use_beam_search, args.disable_tqdm)
    t = time.perf_counter()
    output_list = asyncio.run(client.benchmark(requests_prompts, requests_times))
    benchmark_time = time.perf_counter() - t
    # pylint: disable=line-too-long
    output = {
        "time": benchmark_time,
        "outputs": [request_func_output.model_dump() for request_func_output in output_list],  # type: ignore
        "inputs": requests_prompts,
        "tokenizer": args.tokenizer,
    }
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(json.dumps(output, indent=4))  # type: ignore
    else:
        logger.debug(f"{output_list}")


if __name__ == '__main__':
    main()
