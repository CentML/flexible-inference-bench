import argparse
import asyncio
import time
from transformers import AutoTokenizer
from flexible_inference_benchmark.main import generate_request_times, generate_prompts
from flexible_inference_benchmark.engine.client import Client


def test_backend_function(vllm_server, args_configs):
    base_url = f"http://localhost:{vllm_server[1]}"

    for args_config in args_configs:
        args_config["base_url"] = base_url
        args = argparse.Namespace(**args_config)

        requests_times = generate_request_times(args)
        size = len(requests_times)
        tokenizer_id = args.tokenizer if args.tokenizer else args.model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        requests_prompts = generate_prompts(args, tokenizer, size)
        min_length = min(len(requests_prompts), len(requests_times))
        requests_prompts = requests_prompts[:min_length]
        requests_times = requests_times[:min_length]
 
        args.api_url = f"{args.base_url}{args.endpoint}"
        client = Client(
            args.backend,
            args.api_url,
            args.base_url,
            args.model,
            args.best_of,
            args.use_beam_search,
            True if args.verbose else args.disable_tqdm,
            args.https_ssl,
            not args.disable_ignore_eos,
            not args.disable_stream,
            args.cookies,
            args.verbose,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_concurrent=None,
            wave=None,
            logprobs=None,
        )

        t = time.perf_counter() 
        output_list =  asyncio.run(client.benchmark(requests_prompts, requests_times, requests_media=[[]] * len(requests_prompts)))
        
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
        assert output["stream"] == (not args.disable_stream)
        for item in output["outputs"]:
            assert item["success"]
            assert len(item["itl"]) == 0
            assert item["ttft"] == 0.0
            assert item["latency"] > 0.0
