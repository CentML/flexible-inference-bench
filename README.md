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

### Parameters
| argument | description |
| --- | --- |
| `--seed` |seed for reproducibility |
| `--backend` | backend options: `tgi`,`vllm`,`cserve`,`lmdeploy`,`deepspeed-mii`,`openai`,`openai-chat`,`tensorrt-llm` |
| `--base-url` | Server or API base url, if not using http host and port. |
| `--host_port` | Host and port for the server in **`host:port`** format |
| `--endpoint` | API endpoint |
| one of <br> `--num-of-req` **or** <br> `--max-time-for-reqs` | <br> Total number of requests to send <br> time window for sending requests **(in seconds)**|
| `--request-distribution` | Distribution for sending requests: <br> **eg:** exponential 5 (request will follow an exponential distribution with an average time between requests of **5 seconds**) <br> options: <br> `poisson rate` <br> `uniform min_val max_val` <br> `normal mean std` | 
| `--input-token-distribution` | Request distribution for prompt length. eg: <br> `uniform min_val max_val` <br> `normal mean std` |
| `--output-token-distribution` | Request distribution for output token length. eg: <br> `uniform min_val max_val` <br> `normal mean std` |
| one of:<br>`--prefix-text`<br>`--prefix-len`<br>`--no-prefix` | <br> Text to use as prefix for all requests. <br> Length of prefix to use for all requests. <br> No prefix for requests. |
| `--dataset-name` | Name of the dataset to benchmark on <br> `{sharegpt,other,random}` |
| `--dataset-path` | Path to the dataset |
| `--model` | Name of the model |
| `--tokenizer` | Name or path of the tokenizer, if not using the default tokenizer.
| `--trust-remote-code` | Trust remote code from huggingface |
| `--disable-tqdm` | Specify to disable tqdm progress bar |
| `--best-of` | Number of best completions to return |
| `--use-beam-search` | Use beam search for completions |
| `--output-file` | Output json file to save the results |
| `--debug` | Log debug messages |
| `--config-file` | path to configuration file

**For ease of use we recommend passing a configuration file with all the required parameters for your use case. An example is provided in the file args.json**

### Output
The output json file in an array of objects that contain the following fields:<br>
* `text`: Generated text
* `success`: Whether the request was successful
* `latency`: End-to-end time for the request
* `ttft`: Time to first token
* `itl`: Inter-token latency
* `prompt_len`: Length of the prompt
* `error`: Error message