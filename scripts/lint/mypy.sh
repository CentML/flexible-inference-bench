#!/bin/bash

# work in the same directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $SCRIPT_DIR

python -m mypy \
     --install-types --non-interactive \
     --config-file ./mypy.ini \
     ../../modular_inference_benchmark

