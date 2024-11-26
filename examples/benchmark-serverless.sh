#!/bin/bash

git checkout xinli/serverless
pip install -e .

if [ ! -f ShareGPT_V3_unfiltered_cleaned_split.json ]; then
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for rate in 0.2 0.5 1 5 10 15 20 25 30
do
    fib benchmark \
	    --config-file $SCRIPT_DIR/benchmark-serverless.json \
	    --dataset-pat ShareGPT_V3_unfiltered_cleaned_split.json \
	    --request-distribution poisson $rate \
	    --use-out-token-dist-sharegpt \
	    --output-file fib_bench_serverless_poisson_$rate.json \
	    --num-of-req 200
done
