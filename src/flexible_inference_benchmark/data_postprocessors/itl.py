"""
Simple example of a data postprocessor script with minimal error checking and typing that shows a plot of ITLs.
"""

import json
import argparse
import matplotlib.pyplot as plt


def add_itl_parser(subparsers: argparse._SubParsersAction):
    itl_parser = subparsers.add_parser("generate-itl-plot")
    itl_parser.add_argument("--datapath", type=str, required=True, help="Path to the data file")
    itl_parser.add_argument("--output", type=str, required=False, help="Path to save the plot")
    itl_parser.add_argument('--request-num', type=int, default=0, help='Request number to plot')


def plot_itl(data, idx, output):
    itls = data["outputs"][idx]["itl"]
    plt.title(f"Inter-Token Latencies for Request {idx}")
    plt.xlabel("Token")
    plt.ylabel("Inter-Token Latency (s)")
    plt.scatter(list(range(len(itls))), itls)
    if output:
        plt.savefig(output)
    plt.show()


def run(args: argparse.Namespace):
    with open(args.datapath, 'r') as f:
        data = json.load(f)
    plot_itl(data, args.request_num, args.output)
