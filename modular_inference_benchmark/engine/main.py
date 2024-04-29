import argparse
import logging
import random
import sys
import numpy as np
from engine.distributions import DISTRIBUTION_CLASSES
from utils.utils import configure_logging, get_tokenizer
from engine.data import ShareGPT, Textfile, PREFIX_OPTIONS

logger = logging.getLogger(__name__)


def select_distribution(dist_type: str, rate: float, low: int, high: int, mean: float, std: float):
    if dist_type in ["poisson", "exponential", "even", "same"]:
        return DISTRIBUTION_CLASSES[dist_type](rate)
    elif dist_type == "uniform":
        return DISTRIBUTION_CLASSES[dist_type](low, high)
    else:
        return DISTRIBUTION_CLASSES[dist_type](mean, std)


def generate_request_times(args: argparse.Namespace):
    size = args.num_of_req
    dist = select_distribution(
        args.inc_req_dist, args.rate, args.uniform_low, args.uniform_high, args.normal_mean, args.normal_std
    )
    requests_times = dist.generate_distribution(size)
    return requests_times


def generate_prompts(args: argparse.Namespace):
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer else model_id
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)
    filename = args.dataset_path
    prompt_cls = None
    if args.dataset_name == 'sharegpt':
        logger.info(
            "User selected sharegpt dataset.\nIgnoring prompt and output length distribution and following the shapes from the dataset.\n"
        )
        prompt_cls = ShareGPT(filename, tokenizer)
    else:
        logger.info(
            f"User selected {args.dataset_name} dataset. Generating prompt and output lengths from distributions"
        )
        input_prompt_dist = select_distribution(
            args.prompt_len_dist, args.rate, args.uniform_low, args.uniform_high, args.normal_mean, args.normal_std
        )
        output_token_dist = select_distribution(
            args.output_len_dist, args.rate, args.uniform_low, args.uniform_high, args.normal_mean, args.normal_std
        )

        prompt_cls = Textfile(
            args.dataset_name,
            args.dataset_path,
            args.prompt_prefix,
            args.prefix_text,
            args.prefix_len,
            input_prompt_dist,
            output_token_dist,
            tokenizer,
        )

    if not prompt_cls:
        logger.error("Error generating prompts, exiting benchmark .....")
        sys.exit(1)

    return prompt_cls.generate_data(args.num_of_req)


def parse_args():

    parser = argparse.ArgumentParser(description="CentML Inference Benchmark")

    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")

    parser.add_argument(
        "--backend",
        type=str,
        default='cserve',
        # choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="Backend inference engine.",
    )

    parser.add_argument(
        "--base-url", type=str, default=None, help="Server or API base url if not using http host and port."
    )

    parser.add_argument("--host", type=str, default="localhost")

    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--endpoint", type=str, default="/v1/completions", help="API endpoint.")

    parser.add_argument(
        "--best-of", type=int, default=1, help="Generates `best_of` sequences per prompt and " "returns the best one."
    )

    parser.add_argument("--use-beam-search", action="store_true")

    parser.add_argument(
        "--output-len-dist",
        default="normal",
        choices=list(DISTRIBUTION_CLASSES.keys()),
        help="Distribution of output sequence length.",
    )

    parser.add_argument(
        "--prompt-len-dist",
        default="uniform",
        choices=list(DISTRIBUTION_CLASSES.keys()),
        help="Distribution of propmt length.",
    )

    parser.add_argument(
        "--inc-req-dist",
        default="exponential",
        choices=list(DISTRIBUTION_CLASSES.keys()),
        help="distribution of incoming requests.",
    )

    parser.add_argument(
        "--rate", type=int, default=5, help="rate for exponential, poisson, same or even distributions."
    )

    parser.add_argument("--num-of-req", type=int, default=200, help="total number of request.")

    parser.add_argument("--normal-mean", type=float, default=1, help="mean for normal distribution.")

    parser.add_argument("--normal-std", type=float, default=0, help="standard deviation for normal distributions.")

    parser.add_argument("--uniform-low", type=int, default=100, help="low value for uniform distributions.")

    parser.add_argument("--uniform-high", type=int, default=200, help="high value for uniform distributions.")

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

    args = parser.parse_args()
    return args


def main():
    configure_logging()
    args = parse_args()
    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    requests_times = generate_request_times(args)
    requests_prompts = generate_prompts(args)
    print(requests_times[:3])
    print(requests_prompts[:3])


if __name__ == '__main__':
    main()
