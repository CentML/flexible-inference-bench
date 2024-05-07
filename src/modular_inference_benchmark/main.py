import argparse
import json
import dataclasses
import logging
import random
import asyncio
import sys
from typing import List, Any, Tuple
import numpy as np
from transformers import AutoTokenizer
from modular_inference_benchmark.engine.distributions import DISTRIBUTION_CLASSES, Distribution
from modular_inference_benchmark.utils.utils import configure_logging
from modular_inference_benchmark.engine.data import ShareGPT, Textfile, Random, PREFIX_OPTIONS
from modular_inference_benchmark.engine.client import Client
from modular_inference_benchmark.engine.backend_functions import ASYNC_REQUEST_FUNCS

logger = logging.getLogger(__name__)


def select_distribution(args: List[Any]) -> Distribution:
    dist_type = args[0]
    dist_args = (float(i) for i in args[1:])
    return DISTRIBUTION_CLASSES[dist_type](*dist_args)  # type: ignore


def generate_request_times(args: argparse.Namespace) -> List[Any]:
    size = args.num_of_req
    dist = select_distribution(args.request_distribution)
    requests_times = dist.generate_distribution(size)
    return requests_times


def generate_prompts(args: argparse.Namespace) -> List[Tuple[str, int, int]]:
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    filename = args.dataset_path
    prompt_cls = None
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

        if args.prompt_prefix in ("no-prefix", "prefix-with-len"):
            prefix_len = args.prefix_len if args.prompt_prefix == "prefix-with-len" else 0
            prompt_cls = (
                Random.with_prefix_len(prefix_len, input_prompt_dist, output_token_dist, tokenizer)  # type: ignore
                if args.dataset_name == "random"
                else Textfile.with_prefix_len(filename, prefix_len, input_prompt_dist, output_token_dist, tokenizer)
            )
        else:
            prompt_cls = (
                Random.with_prefix_str(
                    args.prefix_text, input_prompt_dist, output_token_dist, tokenizer  # type: ignore
                )
                if args.dataset_name == "random"
                else Textfile.with_prefix_str(
                    filename, args.prefix_text, input_prompt_dist, output_token_dist, tokenizer
                )
            )

    if not prompt_cls:
        logger.error("Error generating prompts, exiting benchmark .....")
        sys.exit(1)

    factor = 1.2
    size = int(args.num_of_req * factor)

    data = prompt_cls.generate_data(size)
    if len(data) < args.num_of_req:
        logger.warning("The number of requests is less than the size.")
    else:
        data = data[: args.num_of_req]
    return data


def parse_args() -> Any:  # type: ignore

    parser = argparse.ArgumentParser(description="CentML Inference Benchmark")

    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")

    parser.add_argument(
        "--backend",
        type=str,
        default='cserve',
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="Backend inference engine.",
    )

    parser.add_argument(
        "--base-url", type=str, default=None, help="Server or API base url if not using http host and port."
    )

    parser.add_argument("--host", type=str, default="localhost")

    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--endpoint", type=str, default="/v1/completions", help="API endpoint.")

    parser.add_argument("--num-of-req", type=int, default=200, help="total number of request.")

    parser.add_argument(
        "--request-distribution",
        nargs="*",
        default=["exponential", 1],
        help="request distribution [Distribution_type (inputs to distribution)]",
    )

    parser.add_argument(
        "--input-token-distribution",
        nargs="*",
        default=["uniform", 0, 255],
        help="request distribution [Distribution_type (inputs to distribution)]",
    )

    parser.add_argument(
        "--output-token-distribution",
        nargs="*",
        default=["uniform", 0, 255],
        help="request distribution [Distribution_type (inputs to distribution)]",
    )

    parser.add_argument(
        "--prompt-prefix",
        type=str,
        choices=PREFIX_OPTIONS,
        default="no-prefix",
        help="Choose if you would like to share a similar prefix for all the requests.",
    )

    parser.add_argument(
        "--prefix-text", type=str, default="This is a default prompt", help="text to use as prefix for all requests."
    )

    parser.add_argument("--prefix-len", type=int, default=20, help="length of prefix to use for all requests.")

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "other", "random"],
        help="Name of the dataset to benchmark on.",
    )

    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset.")

    parser.add_argument("--model", type=str, required=True, help="Name of the model.")

    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Name or path of the tokenizer, if not using the default tokenizer."
    )

    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code from huggingface")

    parser.add_argument("--disable-tqdm", action="store_true", help="Specify to disable tqdm progress bar.")

    parser.add_argument("--best-of", type=int, default=1, help="Number of best completions to return.")

    parser.add_argument("--use-beam-search", action="store_true", help="Use beam search for completions.")

    parser.add_argument("--output-file", type=str, default=None, help="Output json file to save the results.")

    parser.add_argument("--debug", action="store_true", help="Log debug messages")

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    configure_logging(args)
    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    requests_prompts = generate_prompts(args)
    requests_times = generate_request_times(args)[: len(requests_prompts)]

    if args.base_url is None:
        assert args.host and args.port, "Host and port must be provided if base url is not provided."
        args.api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    else:
        args.api_url = f"{args.base_url}/{args.endpoint}"

    client = Client(args.backend, args.api_url, args.model, args.best_of, args.use_beam_search, args.disable_tqdm)
    output_list = asyncio.run(client.benchmark(requests_prompts, requests_times))

    # pylint: disable=line-too-long
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(
                json.dumps([dataclasses.asdict(request_func_output) for request_func_output in output_list], indent=4)  # type: ignore
            )
    else:
        logger.debug(f"{output_list}")


if __name__ == '__main__':
    main()
