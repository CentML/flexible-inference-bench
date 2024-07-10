import pytest
import subprocess
import time

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
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # wait for vllm to start
    time.sleep(10)
    # check it started successfully
    assert not vllm_proc.poll(), vllm_proc.stdout.read().decode("utf-8")
    yield vllm_proc, port
    # Shut it down at the end of the pytest session
    vllm_proc.terminate()


@pytest.fixture(scope="session")
def args_config():
    args = {
        "backend": "vllm",
        "https_ssl": True,
        "endpoint": "/v1/completions",
        "dataset_name": "random",
        "dataset_path": None,
        "input_token_distribution": ["normal", 200, 10],
        "output_token_distribution": ["uniform", 200, 201],
        "request_distribution": ["exponential", 1],
        # Please, do not increase number of request. Pytest + asyncio hangs with number of request
        # greater than 10
        "num_of_req": 10, 
        "model": "gpt2",
        "tokenizer": "gpt2",
        "no_prefix": True,
        "prefix_text": None,
        "prefix_len": None,
        "disable_ignore_eos": False,
        "disable_stream": True,
        "cookies": {},
        "disable_tqdm":True,
        "best_of":1,
        "use_beam_search": False,
        "debug":False,
        "verbose":False,
    }

    return args
