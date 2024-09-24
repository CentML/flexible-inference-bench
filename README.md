# Flexible Inference Benchmarker
A modular, extensible LLM inference benchmarking framework that supports multiple benchmarking frameworks and paradigms.

This benchmarking framework operates entirely external to any serving framework, and can easily be extended and modified. It is intended to be fully-featured to provide a variety of statistics and profiling modes and be easily extensible.

## Installation
```
cd flexible-inference-benchmark
pip install .
```

## Usage
After installing with the above instructions, the benchmarker can be invoked with `fib <args>`.

After you get your output (using `--output-file`), you can invoke one of the data postprocessors in `data_postprocessors`.

### Parameters
| argument | description |
| --- | --- |
| `--seed` | Seed for reproducibility. |
| `--backend` | Backend options: `tgi`,`vllm`,`cserve`,`cserve-debug`,`lmdeploy`,`deepspeed-mii`,`openai`,`openai-chat`,`tensorrt-llm`. <br> **For tensorrt-llm temperature is set to 0.01 since NGC container >= 24.06 does not support 0.0** |
| `--base-url` | Server or API base url, without endpoint |
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
| `--disable-ignore-eos` | Ignores end of sequence.<br> **Note:** Not valid argument for TensorRT-LLM |
| `--disable-stream` | The requests are send with Stream: False. (Used for APIs without an stream option) |
| `--cookies` | Include cookies in the request. |
| `--config-file` | Path to configuration file. |

**For ease of use we recommend passing a configuration file with all the required parameters for your use case. Examples are provided in `examples/`**

### Output
The output json file in an array of objects that contain the following fields:<br>
* `backend`: backend used
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
Below is a description of the data postprocessors. Each can be invoked with `fib <postprocessor name here> [Options]`

#### `analyse`
Prints the following output for a given run, same as vLLM.

```
============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  4.12      
Total input tokens:                      3978      
Total generated tokens:                  4000      
Request throughput (req/s):              4.85      
Input token throughput (tok/s):          964.98    
Output token throughput (tok/s):         970.32    
---------------Time to First Token----------------
Mean TTFT (ms):                          6.79      
Median TTFT (ms):                        4.81      
P99 TTFT (ms):                           17.90     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.57      
Median TPOT (ms):                        1.59      
P99 TPOT (ms):                           1.90      
---------------Inter-token Latency----------------
Mean ITL (ms):                           1.57      
Median ITL (ms):                         1.47      
P99 ITL (ms):                            2.71      
==================================================
```

Supports the following args:

| argument | description |
| --- | --- |
| `--datapath` | Path to the output json file produced. |

#### `generate-itl-plot`

Returns a plot of inter-token latencies for a specific request. Takes the following args:

| argument | description |
| --- | --- |
| `--datapath` | Path to the output json file produced. |
| `--output` | Path to save figure supported by matplotlib. |
| `--request-num` | Which request to produce ITL plot for. |

#### `generate-ttft-plot`

Generates a simple CDF plot of **time to first token** requests. You can pass a single file or  a list of generated files from the benchmark to make a comparisson <br>

| argument | description |
| --- | --- |
| `--files` | file(s) to generate the plot

## `Example`

Let's take vllm as the backend for our benchmark.
You can install vllm with the command:<br>
`pip install vllm`

We will use gpt2 as the model<br>
`python -m vllm.entrypoints.openai.api_server --model gpt2`

Once the backend is up and running we can go to the examples folder and run the inference benchmark using vllm_args.json file <br>
`cd examples`<br>
`fib benchmark --config-file vllm_args.json --output-file vllm-benchmark.json`

then you can run the performance analysis post-processor:<br>
`fib analyse --datapath vllm-benchmark.json` <br>

```
============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  4.12      
Total input tokens:                      3978      
Total generated tokens:                  4000      
Request throughput (req/s):              4.85      
Input token throughput (tok/s):          964.98    
Output token throughput (tok/s):         970.32    
---------------Time to First Token----------------
Mean TTFT (ms):                          6.79      
Median TTFT (ms):                        4.81      
P99 TTFT (ms):                           17.90     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.57      
Median TPOT (ms):                        1.59      
P99 TPOT (ms):                           1.90      
---------------Inter-token Latency----------------
Mean ITL (ms):                           1.57      
Median ITL (ms):                         1.47      
P99 ITL (ms):                            2.71      
==================================================
```