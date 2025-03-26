# Taken from vLLM benchmarks
import json
import os
import sys
import time
import traceback
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field
import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'


class RequestFuncInput(BaseModel):
    prompt: str
    media: List[str]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False
    ssl: bool = True
    ignore_eos: bool = True
    stream: bool = True
    cookies: Dict[str, str]
    logprobs: Optional[int] = None


class RequestFuncOutput(BaseModel):
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = Field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: Optional[int] = None


async def async_request_tgi(
    idx: int,
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm],
    verbose: bool,
    wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        assert request_func_input.logprobs is None
        params = {
            "best_of": request_func_input.best_of,
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
        }
        payload = {"inputs": request_func_input.prompt, "parameters": params}
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        if verbose:
            print_verbose(idx, request_func_input, st, 0, 0, True)
        try:
            async with session.post(url=api_url, json=payload, verify_ssl=request_func_input.ssl) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]

                    if verbose:
                        print_verbose(idx, request_func_input, 0, most_recent_timestamp, output.latency, False)

        except aiohttp.ClientConnectorError:
            output.success = False
            output.error = "connection error, please verify the server is running"

        except Exception:  # pylint: disable=broad-except
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    idx: int,
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm],
    verbose: bool,
    wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        assert request_func_input.best_of == 1
        assert request_func_input.logprobs is None
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.01,  # does not support 0.0 as of NGC container version 24.06
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        if verbose:
            print_verbose(idx, request_func_input, most_recent_timestamp, 0, 0, True)
        try:
            async with session.post(url=api_url, json=payload, verify_ssl=request_func_input.ssl) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    if verbose:
                        print_verbose(idx, request_func_input, 0, most_recent_timestamp, output.latency, False)

                else:
                    output.error = response.reason or ""
                    output.success = False

        except aiohttp.ClientConnectorError:
            output.success = False
            output.error = "connection error, please verify the server is running"

        except Exception:  # pylint: disable=broad-except
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    idx: int,
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm],
    verbose: bool,
    wait_time: float
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert request_func_input.best_of == 1
        assert not request_func_input.use_beam_search
        assert request_func_input.logprobs is None

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        if verbose:
            print_verbose(idx, request_func_input, st, 0, 0, True)
        try:
            async with session.post(
                url=request_func_input.api_url, json=payload, verify_ssl=request_func_input.ssl
            ) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    rcv_time = time.perf_counter()
                    output.latency = rcv_time - st
                    output.generated_text = parsed_resp["text"][0]
                    output.success = True
                    if verbose:
                        print_verbose(idx, request_func_input, 0, rcv_time, output.latency, False)
                else:
                    output.error = response.reason or ""
                    output.success = False

        except aiohttp.ClientConnectorError:
            output.success = False
            output.error = "connection error, please verify the server is running"

        except Exception:  # pylint: disable=broad-except
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    idx: int,
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm],
    verbose: bool,
    wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("v1/completions"), "OpenAI Completions API URL must end with 'v1/completions'."
    assert not request_func_input.use_beam_search

    if request_func_input.stream:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, cookies=request_func_input.cookies) as session:
            payload = {
                "model": request_func_input.model,
                "prompt": request_func_input.prompt,
                "temperature": 0.0,
                "best_of": request_func_input.best_of,
                "max_tokens": request_func_input.output_len,
                "stream": True,
                "ignore_eos": request_func_input.ignore_eos,
                "stream_options": {"include_usage": True},
            }
            if request_func_input.logprobs is not None:
                payload["logprobs"] = int(request_func_input.logprobs)
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            latency = 0.0
            if verbose:
                print_verbose(idx, request_func_input, st, 0, 0, True)
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers, verify_ssl=request_func_input.ssl
                ) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                            if chunk == "[DONE]":
                                latency = time.perf_counter() - st
                            else:
                                data = json.loads(chunk)

                                if len(data["choices"]) > 0 and data["choices"][0]["text"]:
                                    timestamp = time.perf_counter()
                                    # First token
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    # NOTE: Some completion API might have a last
                                    # usage summary response without a token so we
                                    # do not want to include as inter-token-latency
                                    elif data.get("usage", None) is None:
                                        output.itl.append(timestamp - most_recent_timestamp)

                                    most_recent_timestamp = timestamp
                                    generated_text += data["choices"][0]["text"]

                                if data["usage"]:
                                    output.output_len = data["usage"]["completion_tokens"]

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency

                        if verbose:
                            print_verbose(idx, request_func_input, 0, most_recent_timestamp, latency, False)
                    else:
                        output.success = False
                        output.error = (
                            f"There was an error reaching the endpoint. Error code: {response.status} {response.reason}"
                        )
            except aiohttp.ClientConnectorError:
                output.success = False
                output.error = "connection error, please verify the server is running"

            except Exception:  # pylint: disable=broad-except
                # print(response.status)
                output.success = False
                exc_info = sys.exc_info()
                output.error += "".join(traceback.format_exception(*exc_info))
    else:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, cookies=request_func_input.cookies) as session:
            payload = {
                "model": request_func_input.model,
                "prompt": request_func_input.prompt,
                "temperature": 0.0,
                "best_of": request_func_input.best_of,
                "max_tokens": request_func_input.output_len,
                "ignore_eos": request_func_input.ignore_eos,
                "stream": False,
            }
            if request_func_input.logprobs is not None:
                payload["logprobs"] = int(request_func_input.logprobs)
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len
            output.ttft = 0

            st = time.perf_counter()
            if verbose:
                print_verbose(idx, request_func_input, st, 0, 0, True)
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers, verify_ssl=request_func_input.ssl
                ) as response:
                    if response.status == 200:
                        parsed_resp = await response.json()
                        rcv_time = time.perf_counter()
                        output.latency = rcv_time - st
                        output.generated_text = parsed_resp["choices"][0]["text"]
                        output.success = True
                        if verbose:
                            print_verbose(idx, request_func_input, 0, rcv_time, output.latency, False)
                    else:
                        output.success = False
                        output.error = (
                            f"There was an error reaching the endpoint. Error code: {response.status} {response.reason}"
                        )

            except aiohttp.ClientConnectorError:
                output.success = False
                output.error = "connection error, please verify the server is running"

            except Exception:  # pylint: disable=broad-except
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    idx: int,
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm],
    verbose: bool,
    wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "v1/chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'v1/chat/completions'."

    content_body = [
        {
            "type": "text",
            "text": request_func_input.prompt,
        },
    ]

    for media_item in request_func_input.media:
        content_body.append({
            "type": "image_url",
            "image_url": {
                "url": media_item,
            },
        })

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "messages": [{"role": "user", "content": content_body}],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }
        if request_func_input.logprobs is not None:
            payload["logprobs"] = True
            payload["top_logprobs"] = int(request_func_input.logprobs)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        if verbose:
            print_verbose(idx, request_func_input, st, 0, 0, True)
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers, verify_ssl=request_func_input.ssl
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency

                    if verbose:
                        print_verbose(idx, request_func_input, 0, most_recent_timestamp, output.latency, False)
                else:
                    output.error = response.reason or ""
                    output.success = False

        except aiohttp.ClientConnectorError:
            output.success = False
            output.error = "connection error, please verify the server is running"

        except Exception:  # pylint: disable=broad-except
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_cserve_debug(
    idx: int,
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm],
    verbose: bool,
    wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("v1/generate"), "CServe Completions API URL must end with 'v1/generate'."
    assert not request_func_input.use_beam_search
    assert request_func_input.logprobs is None

    if request_func_input.stream:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, cookies=request_func_input.cookies) as session:
            payload = {
                "prompt": request_func_input.prompt,
                "sampling_params": {"n": 1, "temperature": 0, "max_tokens": request_func_input.output_len},
                "stream": True,
                "ignore_eos": request_func_input.ignore_eos,
            }
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            if verbose:
                print_verbose(idx, request_func_input, st, 0, 0, True)
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers, verify_ssl=request_func_input.ssl
                ) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = chunk_bytes.decode("utf-8")

                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text += chunk

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = time.perf_counter() - st

                        if verbose:
                            print_verbose(idx, request_func_input, 0, most_recent_timestamp, output.latency, False)
                    else:
                        output.success = False
                        output.error = (
                            f"There was an error reaching the endpoint. Error code: {response.status} {response.reason}"
                        )

            except aiohttp.ClientConnectorError:
                output.success = False
                output.error = "connection error, please verify the server is running"

            except Exception:  # pylint: disable=broad-except
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

    else:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, cookies=request_func_input.cookies) as session:
            payload = {
                "prompt": request_func_input.prompt,
                "sampling_params": {
                    "n": 1,
                    "temperature": 0,
                    "max_tokens": request_func_input.output_len,
                    "ignore_eos": request_func_input.ignore_eos,
                },
                "stream": False,
            }
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len
            output.ttft = 0

            st = time.perf_counter()
            if verbose:
                print_verbose(idx, request_func_input, st, 0, 0, True)
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers, verify_ssl=request_func_input.ssl
                ) as response:
                    if response.status == 200:
                        parsed_resp = await response.json()
                        rcv_time = time.perf_counter()
                        output.latency = rcv_time - st
                        output.generated_text = parsed_resp["text"][0]
                        output.success = True
                        if verbose:
                            print_verbose(idx, request_func_input, 0, rcv_time, output.latency, False)
                    else:
                        output.success = False
                        output.error = (
                            f"There was an error reaching the endpoint. Error code: {response.status} {response.reason}"
                        )

            except aiohttp.ClientConnectorError:
                output.success = False
                output.error = "connection error, please verify the server is running"

            except Exception:  # pylint: disable=broad-except
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
    if pbar:
        pbar.update(1)
    return output


# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def print_verbose(
    idx: int, request_func_input: RequestFuncInput, send_time: float, rcv_time: float, latency: float, sending: bool
) -> None:
    if sending:
        print(
            f"{bcolors.OKBLUE}Sending request with id {idx}",
            f"prompt len: {request_func_input.prompt_len}",
            f"decode len: {request_func_input.output_len}",
            f"at time: {send_time}{bcolors.ENDC}",
        )
    else:
        print(
            f"{bcolors.OKGREEN}Response for req {idx}",
            f"with prompt len: {request_func_input.prompt_len}",
            f"with decode len: {request_func_input.output_len}",
            f"has been received at time: {rcv_time}",
            f"with delay: {latency}",
        )


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "cserve-debug": async_request_cserve_debug,
    "cserve-chat": async_request_openai_chat_completions,
    "cserve": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "tensorrt-llm": async_request_trt_llm,
}
