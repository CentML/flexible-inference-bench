#!/bin/bash

additional_opts=${1:-''} # i.e. --check

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

# run black formatter
python -m black \
    --skip-string-normalization \
    --skip-magic-trailing-comma \
    --exclude=".*pb2.*" \
    --line-length 120 \
    $additional_opts \
    ../../src/modular_inference_benchmark
