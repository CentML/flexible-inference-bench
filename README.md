# Modular Inference Benchmarker
A modular, extensible LLM inference benchmarking framework that supports multiple benchmarking frameworks and paradigms.

This benchmarking framework operates entirely external to any serving framework, and can easily be extended and modified. It is intended to be fully-featured to provide a variety of statistics and profiling modes and be easily extensible.

## Installation
```
cd modular-inference-benchmark
pip install .
```

## Usage
After installing with the above instructions, the benchmarker can be invoked with `python -m modular_inference_benchmarker <args>`.

After you get your output (using `--output-file`), you can invoke one of the data postprocessors in `data_postprocessors`.

### Parameters
| argument | description |
| --- | --- |
| `--seed` | Seed for reproducibility. |
| `--backend` | Backend options: `tgi`,`vllm`,`cserve`,`-cserve-debug`,`lmdeploy`,`deepspeed-mii`,`openai`,`openai-chat`,`tensorrt-llm`. |
| `--base-url` | Server or API base url, if not using http host and port. |
| `--host_port` | Host and port for the server in **`host:port`** format. |
| `--endpoint` | API endpoint. |
| one of <br> `--num-of-req` **or** <br> `--max-time-for-reqs` | <br> Total number of requests to send <br> time window for sending requests **(in seconds)** |
| `--request-distribution` | Distribution for sending requests: <br> **eg:** `exponential 5` (request will follow an exponential distribution with an average time between requests of **5 seconds**) <br> options: <br> `poisson rate` <br> `uniform min_val max_val` <br> `normal mean std`. | 
| `--input-token-distribution` | Request distribution for prompt length. eg: <br> `uniform min_val max_val` <br> `normal mean std`. |
| `--output-token-distribution` | Request distribution for output token length. eg: <br> `uniform min_val max_val` <br> `normal mean std`. |
| one of:<br>`--prefix-text`<br>`--prefix-len`<br>`--no-prefix` | <br> Text to use as prefix for all requests. <br> Length of prefix to use for all requests. <br> No prefix for requests. |
| `--dataset-name` | Name of the dataset to benchmark on <br> {`sharegpt`,`other`,`random`}. |
| `--dataset-path` | Path to the dataset. |
| `--model` | Name of the model. |
| `--tokenizer` | Name or path of the tokenizer, if not using the default tokenizer. |
| `--disable-tqdm` | Specify to disable tqdm progress bar. |
| `--best-of` | Number of best completions to return. |
| `--use-beam-search` | Use beam search for completions. |
| `--output-file` | Output json file to save the results. |
| `--debug` | Log debug messages. |
| `--config-file` | Path to configuration file. |

**For ease of use we recommend passing a configuration file with all the required parameters for your use case. Examples are provided in `examples/`**

### Output
The output json file in an array of objects that contain the following fields:<br>
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

### Data Postprocessors
Below is a description of the data postprocessors.

#### `performance.py`
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

Supports the following args:

| argument | description |
| --- | --- |
| `--datapath` | Path to the output json file produced. |

#### `itl.py`

Returns a plot of inter-token latencies for a specific request. Takes the following args:

| argument | description |
| --- | --- |
| `--datapath` | Path to the output json file produced. |
| `--output` | Path to save figure supported by matplotlib. |
| `--request-num` | Which request to produce ITL plot for. |