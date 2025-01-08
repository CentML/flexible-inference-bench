#!/bin/bash

# Make sure OpenAI API is present
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "Environment variable OPENAI_API_KEY is not set, please set it to use CentML Platform's OpenAI Compatible API endpoint"
    echo "Obtain your API key following this URL: https://app.centml.com/user/credentials"
    exit -1 
fi

pip install -e .

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ulimit -n 100000

for rate in 0.2 0.5 1 2 5 10
do
    if (( $(echo "$rate >= 2" | bc -l) )); then
        num_of_req=500
    else
        num_of_req=100
    fi
    echo "Starting benchmark with request rate: $rate for $num_of_req requests"
    datafile=fib_bench_serverless_poisson_$rate.json
    fib benchmark \
	    --config-file $SCRIPT_DIR/benchmark-serverless.json \
	    -rps $rate \
	    --output-file $datafile \
	    --num-of-req $num_of_req
    if [ $? -ne 0 ]; then
        echo "fib command failed, exiting benchmarking"
        exit -1
    fi
done
