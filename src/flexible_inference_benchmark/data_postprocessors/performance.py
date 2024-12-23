"""
Gets the performance based on vLLM performance with minimal error checking and typing.
"""

import json
import argparse
import numpy as np
from transformers import AutoTokenizer


def add_performance_parser(subparsers: argparse._SubParsersAction) -> None:
    performance_parser = subparsers.add_parser('analyse', help="Summarize the performance of a benchmark record")
    performance_parser.add_argument("datapath", type=str, help='Path to the json file')


def calculate_metrics(input_requests, outputs, benchmark_duration, tokenizer, stream):
    actual_output_lens = []
    total_input = 0
    completed = 0
    itls = []
    tpots = []
    ttfts = []
    for i in range(len(outputs)):
        if outputs[i]["success"]:
            if "output_len" in outputs[i]:
                output_len = outputs[i]["output_len"]
            else:
                output_len  = len(tokenizer(outputs[i]["generated_text"], add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append((outputs[i]["latency"] - outputs[i]["ttft"]) / (output_len - 1))
            itls += outputs[i]["itl"]
            ttfts.append(outputs[i]["ttft"])
            completed += 1
        else:
            actual_output_lens.append(0)

    total_output = sum(actual_output_lens)
    request_throughput = completed / benchmark_duration
    input_throughput = total_input / benchmark_duration
    output_throughput = total_output / benchmark_duration
    mean_ttft_ms = np.mean(ttfts or 0) * 1000
    median_ttft_ms = np.median(ttfts or 0) * 1000
    p99_ttft_ms = np.percentile(ttfts or 0, 99) * 1000
    mean_tpot_ms = np.mean(tpots or 0) * 1000
    median_tpot_ms = np.median(tpots or 0) * 1000
    p99_tpot_ms = np.percentile(tpots or 0, 99) * 1000
    mean_itl_ms = np.mean(itls or 0) * 1000
    median_itl_ms = np.median(itls or 0) * 1000
    p99_itl_ms = np.percentile(itls or 0, 99) * 1000

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", output_throughput))

    if stream:
        print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", mean_ttft_ms))
        print("{:<40} {:<10.2f}".format("Median TTFT (ms):", median_ttft_ms))
        print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", p99_ttft_ms))
        print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", mean_tpot_ms))
        print("{:<40} {:<10.2f}".format("Median TPOT (ms):", median_tpot_ms))
        print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", p99_tpot_ms))
        print("{s:{c}^{n}}".format(s='Inter-token Latency', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean ITL (ms):", mean_itl_ms))
        print("{:<40} {:<10.2f}".format("Median ITL (ms):", median_itl_ms))
        print("{:<40} {:<10.2f}".format("P99 ITL (ms):", p99_itl_ms))
        print("=" * 50)


def run(args: argparse.Namespace):
    with open(args.datapath, 'r') as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(data["tokenizer"])
    calculate_metrics(data["inputs"], data["outputs"], data["time"], tokenizer, data["stream"])
