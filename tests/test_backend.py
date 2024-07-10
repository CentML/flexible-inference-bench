import argparse
import asyncio
import time
from flexible_inference_benchmark.main import generate_request_times, generate_prompts
from flexible_inference_benchmark.engine.client import Client


def test_backend_function(vllm_server, args_config):
    ###############################################################
    ## STREAM DISABLED
    ###############################################################
    args_config["base_url"] = f"http://localhost:{vllm_server[1]}"
    args = argparse.Namespace(**args_config)

    requests_times = generate_request_times(args)
    size = len(requests_times)
    requests_prompts = generate_prompts(args, size)
    min_length = min(len(requests_prompts), len(requests_times))
    requests_prompts = requests_prompts[:min_length]
    requests_times = requests_times[:min_length]
 
    args.api_url = f"{args.base_url}{args.endpoint}"
    client = Client(
        args.backend,
        args.api_url,
        args.model,
        args.best_of,
        args.use_beam_search,
        True if args.verbose else args.disable_tqdm,
        args.https_ssl,
        not args.disable_ignore_eos,
        not args.disable_stream,
        args.cookies,
        args.verbose,
    )

    t = time.perf_counter() 
    output_list =  asyncio.run(client.benchmark(requests_prompts, requests_times))
    
    benchmark_time = time.perf_counter() - t
    
    output = {
        "backend": args.backend,
        "time": benchmark_time,
        "outputs": [request_func_output.model_dump() for request_func_output in output_list],
        "inputs": requests_prompts,
        "tokenizer": args.tokenizer if args.tokenizer else args.model,
        "stream": not args.disable_stream,
    }
    
    assert len(output["outputs"]) == args.num_of_req
    assert output["stream"] == False
    for item in output["outputs"]:
        if item["success"] == True:
            assert len(item["itl"]) == 0
            assert item["ttft"] == 0.0
            assert item["latency"] > 0.0

    #########################################################################
    ##  STREAM ENABLED
    #########################################################################

    args_config["base_url"] = f"http://localhost:{vllm_server[1]}"
    args_config["disable_stream"] = False
    args = argparse.Namespace(**args_config)

    requests_times = generate_request_times(args)
    size = len(requests_times)
    requests_prompts = generate_prompts(args, size)
    min_length = min(len(requests_prompts), len(requests_times))
    requests_prompts = requests_prompts[:min_length]
    requests_times = requests_times[:min_length]
 
    args.api_url = f"{args.base_url}{args.endpoint}"
    client = Client(
        args.backend,
        args.api_url,
        args.model,
        args.best_of,
        args.use_beam_search,
        True if args.verbose else args.disable_tqdm,
        args.https_ssl,
        not args.disable_ignore_eos,
        not args.disable_stream,
        args.cookies,
        args.verbose,
    )

    t = time.perf_counter() 
    output_list =  asyncio.run(client.benchmark(requests_prompts, requests_times))
    
    benchmark_time = time.perf_counter() - t
    
    output = {
        "backend": args.backend,
        "time": benchmark_time,
        "outputs": [request_func_output.model_dump() for request_func_output in output_list],  # type: ignore
        "inputs": requests_prompts,
        "tokenizer": args.tokenizer if args.tokenizer else args.model,
        "stream": not args.disable_stream,
    }
    
    assert len(output["outputs"]) == args.num_of_req
    assert output["stream"] == True
    for item in output["outputs"]:
        if item["success"] == True:
            assert len(item["itl"]) > 0
            assert item["ttft"] > 0.0
            assert item["latency"] > 0.0

