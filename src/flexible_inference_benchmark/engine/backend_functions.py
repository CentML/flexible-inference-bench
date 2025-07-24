# Taken from vLLM benchmarks
# pylint: disable=too-many-positional-arguments
import json
import os
import sys
import time
import traceback
from typing import List, Optional, Dict, Any
from contextlib import nullcontext
from pydantic import BaseModel, Field
import aiohttp
from tqdm.asyncio import tqdm
from opentelemetry import trace
from flexible_inference_benchmark.utils.telemetry import create_span_attributes

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

# Get the tracer
tracer = trace.get_tracer(__name__)


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
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    run_id: Optional[str] = None
    json_response: bool = False


class RequestFuncOutput(BaseModel):
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = Field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: Optional[int] = None


def apply_sampling_params(
    payload: Any,
    request_func_input: RequestFuncInput,
    always_top_p: bool = True,
    temp_min: float = 0.0,
    top_p_max: float = 1.0,
) -> None:
    """
    Apply sampling parameters to the payload.
    """
    payload["temperature"] = max(temp_min, request_func_input.temperature)
    if request_func_input.top_p is not None:
        payload["top_p"] = min(request_func_input.top_p, top_p_max)
    elif always_top_p:
        payload["top_p"] = top_p_max
    if request_func_input.top_k is not None:
        payload["top_k"] = request_func_input.top_k
    if request_func_input.best_of > 1:
        payload["best_of"] = request_func_input.best_of


async def async_request_tgi(
    idx: int, request_func_input: RequestFuncInput, pbar: Optional[tqdm], verbose: bool, wait_time: float
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
        }
        apply_sampling_params(params, request_func_input, temp_min=0.01, top_p_max=0.99)
        payload = {"inputs": request_func_input.prompt, "parameters": params}
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        if verbose:
            print_verbose(idx, request_func_input, st, 0, 0, True)
        try:
            async with session.post(url=api_url, json=payload, ssl=request_func_input.ssl) as response:
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
    idx: int, request_func_input: RequestFuncInput, pbar: Optional[tqdm], verbose: bool, wait_time: float
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
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        # does not support temp 0.0 as of NGC container version 24.06
        apply_sampling_params(payload, request_func_input, temp_min=0.01)
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        if verbose:
            print_verbose(idx, request_func_input, most_recent_timestamp, 0, 0, True)
        try:
            async with session.post(url=api_url, json=payload, ssl=request_func_input.ssl) as response:
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
    idx: int, request_func_input: RequestFuncInput, pbar: Optional[tqdm], verbose: bool, wait_time: float
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert request_func_input.best_of == 1
        assert not request_func_input.use_beam_search
        assert request_func_input.logprobs is None
        payload = {"prompt": request_func_input.prompt, "max_tokens": request_func_input.output_len}
        apply_sampling_params(payload, request_func_input, temp_min=0.01)
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
                url=request_func_input.api_url, json=payload, ssl=request_func_input.ssl
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
    idx: int, request_func_input: RequestFuncInput, pbar: Optional[tqdm], verbose: bool, wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("v1/completions"), "OpenAI Completions API URL must end with 'v1/completions'."
    assert not request_func_input.use_beam_search

    if request_func_input.stream:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, cookies=request_func_input.cookies) as session:
            payload = {
                "model": request_func_input.model,
                "prompt": request_func_input.prompt,
                "best_of": request_func_input.best_of,
                "max_tokens": request_func_input.output_len,
                "stream": True,
                "ignore_eos": request_func_input.ignore_eos,
                "stream_options": {"include_usage": True},
            }
            apply_sampling_params(payload, request_func_input, always_top_p=False)
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
                    url=api_url, json=payload, headers=headers, ssl=request_func_input.ssl
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

                                if len(data["choices"]) > 0 and data["choices"][0]["text"] is not None:
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
                                    if "completion_tokens" in data["usage"]:
                                        output.output_len = int(data["usage"]["completion_tokens"])
                                    if "prompt_tokens" in data["usage"]:
                                        output.prompt_len = int(data["usage"]["prompt_tokens"])

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
                "best_of": request_func_input.best_of,
                "max_tokens": request_func_input.output_len,
                "ignore_eos": request_func_input.ignore_eos,
                "stream": False,
            }
            apply_sampling_params(payload, request_func_input, always_top_p=False)
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
                    url=api_url, json=payload, headers=headers, ssl=request_func_input.ssl
                ) as response:
                    if response.status == 200:
                        parsed_resp = await response.json()
                        rcv_time = time.perf_counter()
                        output.latency = rcv_time - st
                        output.generated_text = parsed_resp["choices"][0]["text"]
                        output.success = True
                        if verbose:
                            print_verbose(idx, request_func_input, 0, rcv_time, output.latency, False)
                        if parsed_resp.get("usage", None):
                            if "completion_tokens" in parsed_resp["usage"]:
                                output.output_len = int(parsed_resp["usage"]["completion_tokens"])
                            if "prompt_tokens" in parsed_resp["usage"]:
                                output.prompt_len = int(parsed_resp["usage"]["prompt_tokens"])
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
    idx: int, request_func_input: RequestFuncInput, pbar: Optional[tqdm], verbose: bool, wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "v1/chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'v1/chat/completions'."

    content_body: List[dict[str, Any]] = [{"type": "text", "text": request_func_input.prompt}]

    for media_item in request_func_input.media:
        content_body.append({"type": "image_url", "image_url": {"url": media_item}})

    telemetry_enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    otel_span = (
        tracer.start_as_current_span(
            f"request_{idx}",
            attributes=create_span_attributes(
                prompt_tokens=request_func_input.prompt_len,
                image_count=len(request_func_input.media) if request_func_input.media else 0,
                image_sizes=[len(img) for img in request_func_input.media] if request_func_input.media else [],
                response_tokens=0,  # Will be updated after response
                run_id=request_func_input.run_id or "unknown",  # Provide default value for None
            ),
        )
        if telemetry_enabled
        else nullcontext()
    )
    with otel_span as span:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            assert not request_func_input.use_beam_search

            # Apply JSON response formatting if flag is enabled
            if request_func_input.json_response:
                append_msg = (
                    "\nPlease send your response as a JSON object. "
                    "Follow this schema: {'assistant_response': 'your full, detailed response here'}. "
                    "Do not include any other text or formatting. "
                    "Only return the JSON object without any additional text or explanation."
                )
                if isinstance(content_body, str):
                    content_body += append_msg
                else:
                    content_body[-1]["text"] += append_msg

            payload = {
                "model": request_func_input.model,
                "messages": [{"role": "user", "content": content_body}],
                "max_tokens": request_func_input.output_len,
                "stream": True,
                "ignore_eos": request_func_input.ignore_eos,
                "stream_options": {"include_usage": True},
            }

            # Add JSON response format if flag is enabled
            if request_func_input.json_response:
                payload["response_format"] = {"type": "json_object"}
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            apply_sampling_params(payload, request_func_input, always_top_p=False)
            if request_func_input.logprobs is not None:
                payload["logprobs"] = True
                payload["top_logprobs"] = int(request_func_input.logprobs)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            }

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            if verbose:
                print_verbose(idx, request_func_input, st, 0, 0, True)
            try:
                tracer_http_request = (
                    tracer.start_as_current_span("http_request") if telemetry_enabled else nullcontext()
                )
                with tracer_http_request:
                    async with session.post(
                        url=api_url, json=payload, headers=headers, ssl=request_func_input.ssl
                    ) as response:
                        if response.status == 200:
                            latency = 0.0
                            response_success_span = (
                                tracer.start_as_current_span("response_processing")
                                if telemetry_enabled
                                else nullcontext()
                            )
                            with response_success_span as process_span:
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

                                        delta = data["choices"][0]["delta"] if len(data["choices"]) > 0 else None
                                        content = delta.get("content", None) if delta is not None else None
                                        reasoning_content = (
                                            delta.get("reasoning_content", None) if delta is not None else None
                                        )
                                        if (content is not None or reasoning_content is not None) and not (
                                            ttft == 0.0 and (content == '' or reasoning_content == '')
                                        ):
                                            if ttft == 0.0:
                                                ttft = time.perf_counter() - st
                                                output.ttft = ttft
                                                if process_span:
                                                    process_span.set_attribute("fib.time_to_first_token", ttft)

                                            else:
                                                output.itl.append(timestamp - most_recent_timestamp)
                                                if process_span:
                                                    process_span.set_attribute(
                                                        "fib.inter_token_latency", timestamp - most_recent_timestamp
                                                    )
                                            if content:
                                                generated_text += content
                                            elif reasoning_content:
                                                generated_text += reasoning_content
                                            most_recent_timestamp = timestamp

                                        if "usage" in data:
                                            if data["usage"]["completion_tokens"]:
                                                output.output_len = int(data["usage"]["completion_tokens"])
                                                if process_span:
                                                    process_span.set_attribute(
                                                        "fib.completion_tokens", output.output_len
                                                    )
                                            if data["usage"]["prompt_tokens"]:
                                                output.prompt_len = int(data["usage"]["prompt_tokens"])
                                                if process_span:
                                                    process_span.set_attribute("fib.prompt_tokens", output.prompt_len)

                            output.generated_text = generated_text
                            output.success = True
                            output.latency = latency
                            if span:
                                span.set_attribute("fib.total_latency", latency)
                                span.set_attribute("fib.total_tokens", len(generated_text))

                            if verbose:
                                print_verbose(idx, request_func_input, 0, most_recent_timestamp, output.latency, False)
                        else:
                            output.error = response.reason or ""
                            output.success = False
                            if span:
                                span.set_attribute("fib.error", output.error)

            except aiohttp.ClientConnectorError:
                output.success = False
                output.error = "connection error, please verify the server is running"
                if span:
                    span.set_attribute("fib.error", output.error)

            except Exception:  # pylint: disable=broad-except
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                if span:
                    span.set_attribute("fib.error", output.error)

            if pbar:
                pbar_span = tracer.start_as_current_span("progress_update") if telemetry_enabled else nullcontext()
                with pbar_span:
                    pbar.update(1)

            return output


async def async_request_cserve_debug(
    idx: int, request_func_input: RequestFuncInput, pbar: Optional[tqdm], verbose: bool, wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("v1/generate"), "CServe Completions API URL must end with 'v1/generate'."
    assert not request_func_input.use_beam_search
    assert request_func_input.logprobs is None

    if request_func_input.stream:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, cookies=request_func_input.cookies) as session:
            payload = {
                "prompt": request_func_input.prompt,
                "sampling_params": {"n": 1, "max_tokens": request_func_input.output_len},
                "stream": True,
                "ignore_eos": request_func_input.ignore_eos,
            }
            apply_sampling_params(payload["sampling_params"], request_func_input, always_top_p=False)
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
                    url=api_url, json=payload, headers=headers, ssl=request_func_input.ssl
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
                    "max_tokens": request_func_input.output_len,
                    "ignore_eos": request_func_input.ignore_eos,
                },
                "stream": False,
            }
            apply_sampling_params(payload["sampling_params"], request_func_input, always_top_p=False)
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len
            output.ttft = 0

            st = time.perf_counter()
            if verbose:
                print_verbose(idx, request_func_input, st, 0, 0, True)
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers, ssl=request_func_input.ssl
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


async def async_request_profiler(
    idx: int, request_func_input: RequestFuncInput, pbar: Optional[tqdm], verbose: bool, wait_time: float
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("start_profile") or api_url.endswith(
        "stop_profile"
    ), "Torch Profiler API URL must end with 'start_profile' or 'stop_profile'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "messages": [],
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

        try:
            async with session.post(
                url=api_url, json=payload, headers=headers, verify_ssl=request_func_input.ssl
            ) as response:
                if response.status == 200:
                    output.success = True
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
    "profiler": async_request_profiler,
}
