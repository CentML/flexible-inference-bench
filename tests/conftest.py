import pytest
import subprocess
import time
import requests

from utils import get_open_port


@pytest.fixture(scope="function")
def vllm_server():
    port = get_open_port()
    vllm_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--gpu-memory-utilization",
            "0.3",
            "--model",
            "gpt2",
            "--enforce-eager",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # wait for child process to start
    time.sleep(10)
    # check it started successfully
    assert not vllm_proc.poll(), vllm_proc.stdout.read().decode("utf-8")

    # Wait for up to 60 seconds for the serving engine to initialize
    start_time = time.time()
    while True:
        try:
            response = requests.get(f"http://localhost:{port}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        if time.time() - start_time > 60:
            raise TimeoutError("Server did not start in time")
        time.sleep(0.5)
    
    yield vllm_proc, port
    # Shut it down at the end of the pytest session
    vllm_proc.terminate()


@pytest.fixture(scope="session")
def args_configs():
    common_args = {
        "https_ssl": True,
        "request_distribution": ["exponential", 1],
        "num_of_req": 5,  # Please, do not increase number of request. Pytest + asyncio hangs with total number of request greater than 10
        "model": "gpt2",
        "tokenizer": "gpt2",
        "prefix_text": None,
        "prefix_len": None,
        "disable_ignore_eos": False,
        "disable_stream": True,
        "cookies": {},
        "disable_tqdm": True,
        "best_of": 1,
        "use_beam_search": False,
        "debug": False,
        "verbose": False,
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
    }

    sharegpt_sample_data_path = "tests/data/sharegpt_sample_test_data.json"

    # Send 5 requests with random dataset, streaming disabled, with a prefix string, on the completions backend
    all_configs = []
    all_configs.append({
        "backend": "openai",
        "endpoint": "/v1/completions",
        "dataset_name": "random",
        "dataset_path": None,
        "input_token_distribution": None,
        "output_token_distribution": None,
        "ignore_input_distribution": False,
        "native_output_len": False,
        "prefix_text": "Hello world!",
        **common_args,
    })

    # Send 5 requests with sharegpt dataset, streaming enabled, with forced output distribution and sampling params
    all_configs.append({
        "backend": "openai",
        "endpoint": "/v1/completions",
        "dataset_name": "sharegpt",
        "dataset_path": sharegpt_sample_data_path,
        "input_token_distribution": None,
        "output_token_distribution": ["same", 10],
        "ignore_input_distribution": False,
        "native_output_len": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        **common_args,
    })

    return all_configs
