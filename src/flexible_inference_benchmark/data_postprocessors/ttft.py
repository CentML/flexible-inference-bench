"""
Simple example of a data postprocessor script with minimal error checking and typing that shows a plot of TTFT.
"""

import argparse
import json
import matplotlib.pyplot as plt
import random


def color_scheme_generator(num_colors):
    res = []
    for _ in range(num_colors):
        color = random.randrange(0, 2**24)
        hex_color = hex(color)
        padding = 6 - len(hex_color[2:])
        str_color = "#" + hex_color[2:] + "0" * padding
        res.append(str_color)

    return res


def generate_plot(name, data, color, axis):
    axis.ecdf(data, orientation="horizontal", color=color)
    axis.set_xlabel('CDF')
    axis.set_ylabel('time (sec)')

    ax2 = axis.twiny()
    ax2.hist(data, orientation="horizontal", bins=len(data) // 2, fill=False, edgecolor=color, label=name)
    ax2.set_xticks([])


def plot_ttft(files, color_scheme):
    fig, ax1 = plt.subplots()
    for i, file in enumerate(files):
        with open(file, "r") as f:
            data = json.load(f)
        ttft_arr = [item["ttft"] for item in data["outputs"]]
        generate_plot(data["backend"], ttft_arr, color_scheme[i], ax1)

    fig.tight_layout()
    plt.title('TTFT')
    plt.tight_layout()
    plt.savefig("ttft.pdf")


def add_ttft_parser(subparsers: argparse._SubParsersAction):
    ttft_parser = subparsers.add_parser("generate-ttft-plot")
    ttft_parser.add_argument("--files", nargs="+", help="list of json files")


def run(args: argparse.Namespace):
    color_scheme = color_scheme_generator(len(args.files))
    plot_ttft(args.files, color_scheme)
