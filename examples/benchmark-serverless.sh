#!/bin/bash

pip install -e .

if [ ! -f ShareGPT_V3_unfiltered_cleaned_split.json ]; then
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ulimit -n 100000
# Make sure OpenAI API is present
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "OPENAI_API_KEY is not set, please set it to use CentML Platform's OpenAI Compatible API endpoint"
    exit -1 
fi

for rate in 0.1 0.2 0.5 1 2 5 
do
    if (( $(echo "$rate >= 1" | bc -l) )); then
        num_of_req=1000
    else
        num_of_req=200
    fi
    echo "Starting benchmark with request rate: $rate for $num_of_req requests"
    datafile=fib_bench_serverless_poisson_$rate.json
    fib benchmark \
	    --config-file $SCRIPT_DIR/benchmark-serverless.json \
	    --dataset-pat ShareGPT_V3_unfiltered_cleaned_split.json \
	    --request-distribution poisson $rate \
	    --use-out-token-dist-sharegpt \
	    --output-file $datafile \
	    --num-of-req $num_of_req
    
    echo "Analysing benchmarking data at "
    fib analyse --datapath $datafile
done
