# Flexible Inference Benchmarker
A modular, extensible LLM inference benchmarking framework that supports multiple benchmarking frameworks and paradigms.

This benchmarking framework operates entirely externally to any serving framework, and can easily be extended and modified. It is intended to be fully-featured to provide a variety of statistics and profiling modes.

## Installation
```
cd flexible-inference-bench
pip install .
```

## OpenTelemetry Integration
The framework supports OpenTelemetry integration for distributed tracing. To enable it:

1. Set the following environment variables:
   ```bash
   export OTEL_ENABLED=true
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://your-collector:4317  # Optional, defaults to localhost:4317
   ```

2. Each request will be traced with the following attributes (all prefixed with 'fib.' for easy identification):
   - `fib.prompt.tokens`: Number of tokens in the input prompt
   - `fib.image.count`: Number of images in the request
   - `fib.image.sizes`: List of image sizes in bytes
   - `fib.response.tokens`: Number of tokens in the response
   - `fib.run.id`: Unique identifier for grouping requests from the same benchmark run

3. The traces can be viewed in any OpenTelemetry-compatible backend (e.g., Jaeger, Zipkin, etc.)

## Usage
After installing with the above instructions, the benchmarker can be invoked with `fib benchmark [options]`.

After benchmarking, the results are saved to `output-file.json` (or specified by `--output-file`) and can be postprocessed using `fib analyse <file> | fib generate-ttft-plot [options] | fib generate-itl-plot [options]`.

### Parameters for fib benchmark
| argument | description |
| --- | --- |
| `--seed` | Seed for reproducibility. |
| `--backend` (`-b`) | Backend options: `tgi`,`vllm`,`cserve`,`cserve-debug`,`lmdeploy`,`deepspeed-mii`,`openai`,`openai-chat`,`tensorrt-llm`. <br> **For tensorrt-llm temperature is set to 0.01 since NGC container >= 24.06 does not support 0.0** |
| `--base-url` | Server or API base url, without endpoint. |
| `--endpoint` | API endpoint path. |
| one of <br> `--num-of-req` (`-n`) **or** <br> `--max-time-for-reqs` (`--timeout`) | <br> Total number of requests to send <br> time window for sending requests **(in seconds)** |
| `--request-distribution` | Distribution for sending requests: <br> **eg:** `exponential 5` (request will follow an exponential distribution with an average time between requests of **5 seconds**) <br> options: <br> `poisson rate` <br> `uniform min_val max_val` <br> `normal mean std`. |
| `--request-rate` (`-rps`) | Sets the request distribution to `poisson N`, such that approximately N requests are sent per second. |
| `--max-concurrent` | Limits the number of concurrent in-flight requests to at most N requests. |
| `--input-token-distribution` | Request distribution for prompt length. eg: <br> `uniform min_val max_val` <br> `normal mean std`. |
| `--output-token-distribution` | Request distribution for output token length. eg: <br> `uniform min_val max_val` <br> `normal mean std`. |
| `--varying-requests` (`--wave`) | Sends requests to maintain an oscillating request concurrency given min, max, and sustain. <br> eg: `-n 400 --wave 10 30 20`, will send 400 requests with concurrency between 10 and 30 requests, sustaining for the duration of 20 requests at concurrency extrema. See [this graph](https://drive.google.com/file/d/1c0qrWa3DIEGxJdZO7hcKmehrPreBl84V/view?usp=sharing) for a visual. |
| `--workload` (`-w`) | One of a few presets that define the input and output token distributions for common use-cases. |
| `--num-of-imgs-per-req` | Number of images to attach to each request. Example: `3`. |
| `--img-ratios-per-req` | Image aspect ratios (width x height) to attach per request. Example: `1000x1000`. Default: `500x500` |
| `--img-base-path` | Base image directory. Example: `/path/to/imgs/`. If provided, images will be downloaded to this directory before benchmarking and fed from here. If not provided, images will be fed online, which could cause excessive network delays in large numbers. To enable this, the serving engine also needs to start with the `--allowed-local-media-path /path/to/imgs/` option. Default: `None`. |
| `--num-validation-reqs` | Number of requests to send for validation and warmup before the benchmark. Default: `1`. |
| one of:<br>`--prefix-text` or <br>`--prefix-len` | <br> Text to use as prefix for all requests. <br> Length of prefix to use for all requests. If neither are provided, no prefix is used. |
| `--dataset-name` (`--dataset`) | Name of the dataset to benchmark on <br> {`sharegpt`,`other`,`random`}. |
| `--dataset-path` | Path to the dataset. If `sharegpt` is the dataset and this is not provided, it will be automatically downloaded and cached. Otherwise, the dataset name will default to `other`. |
| `--model` (`-m`) | Name of the model. |
| `--tokenizer` | Name or path of the tokenizer, if not using the default tokenizer. |
| `--disable-tqdm` | Specify to disable tqdm progress bar. |
| `--best-of` | Number of best completions to return. |
| `--use-beam-search` | Use beam search for completions. |
| `--output-file` | Output json file to save the results. |
| `--debug` | Log debug messages. |
| `--profile` | Use Torch Profiler. The endpoint must be launched with VLLM_TORCH_PROFILER_DIR to enable profiler. |
| `--verbose` | Summarize each request. |
| `--disable-ignore-eos` | Ignores end of sequence.<br> **Note:** Not valid argument for TensorRT-LLM |
| `--disable-stream` | The requests are send with Stream: False. (Used for APIs without an stream option) |
| `--cookies` | Include cookies in the request. |
| `--config-file` | Path to configuration file. |
| `--logprobs` | Number of logprobs to return with the request. FIB will not process them, but still useful for measuring the cost of computing / communicating logprobs. Defaults to None. |
| `--temperature` (`--temp`) | Temperature to use for sampling. Defaults to 0.0. |
| `--top-p` | Top-P to use for sampling. Defaults to None, or 1.0 for backends which require it to be specified. |
| `--top-k` | Top-K to use for sampling. Defaults to None. |

In addition to providing these arguments on the command-line, you can use `--config-file` to pre-define the parameters for your use case. Examples are provided in `examples/`

### Output

In addition to printing the analysis results (which can be reproduced using `fib analyse`), the following output artifact is generated:

The output json file contains metadata and a list of request input and output descriptions:<br>
* `backend`: Backend used
* `time`: Total time
* `outputs`: 
    * `text`: Generated text
    * `success`: Whether the request was successful
    * `latency`: End-to-end time for the request
    * `ttft`: Time to first token
    * `itl`: Inter-token latency
    * `prompt_len`: Length of the prompt
    * `error`: Error message
* `inputs`: List of `[prompt string, input tokens, expected output tokens]`
* `tokenizer`: Tokenizer name
* `stream`: Indicates if we used the stream argument or not

### Data Postprocessors
Below is a description of the data postprocessors.

#### `fib analyse <path_to_file>`
Prints the following output for a given run, same as vLLM.

```
============ Serving Benchmark Result ============
Successful requests:                     20
Benchmark duration (s):                  19.39
Total input tokens:                      407
Total generated tokens:                  5112
Request throughput (req/s):              1.03
Input token throughput (tok/s):          20.99
Output token throughput (tok/s):         263.66
---------------Time to First Token----------------
Mean TTFT (ms):                          24.66
Median TTFT (ms):                        24.64
P99 TTFT (ms):                           34.11
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2295.86
Median TPOT (ms):                        2362.54
P99 TPOT (ms):                           2750.76
==================================================
```

#### `fib generate-itl-plot`

Returns a plot of inter-token latencies for a specific request. Takes the following args:

| argument | description |
| --- | --- |
| `--datapath` | Path to the output json file produced. |
| `--output` | Path to save figure supported by matplotlib. |
| `--request-num` | Which request to produce ITL plot for. |

#### `fib generate-ttft-plot`

Generates a simple CDF plot of **time to first token** requests. You can pass a single file or  a list of generated files from the benchmark to make a comparisson <br>

| argument | description |
| --- | --- |
| `--files` | file(s) to generate the plot

## `Example`

Let's take vllm as the backend for our benchmark.
You can install vllm with the command:<br>
`pip install vllm`

We will use gpt2 as the model<br>
`vllm serve gpt2`

And now we can run the benchmark in the CLI:

```
fib benchmark -n 500 -rps inf -w summary
```

Alternatively we can go to the examples folder and run the inference benchmark using a config file: 
```
cd examples
fib benchmark --config-file summary_throughput.json --output-file vllm-benchmark.json
```

```
============ Serving Benchmark Result ============
Successful requests:                     497      
Benchmark duration (s):                  5.09     
Total input tokens:                      58605    
Total generated tokens:                  126519   
Request throughput (req/s):              97.66    
Input token throughput (tok/s):          11516.12 
Output token throughput (tok/s):         24861.49 
---------------Time to First Token----------------
Mean TTFT (ms):                          1508.38  
Median TTFT (ms):                        372.63   
P99 TTFT (ms):                           2858.80  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.34     
Median TPOT (ms):                        9.39     
P99 TPOT (ms):                           10.23    
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.35     
Median ITL (ms):                         8.00     
P99 ITL (ms):                            89.88    
==================================================
```