# pylint: disable=too-many-positional-arguments
import asyncio
import logging
import uuid
from typing import List, Tuple, Callable, Optional, Any, Coroutine, Union, Dict
import random
from tqdm import tqdm
from flexible_inference_benchmark.engine.backend_functions import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)

logger = logging.getLogger(__name__)


class WaveState:
    def __init__(self) -> None:
        # -1: decreasing, 0: sustain, 1: increasing
        self.delta = 1
        self.num_running_requests = 0
        self.num_finished_requests = 0


class Client:
    def __init__(
        self,
        backend: str,
        api_url: str,
        base_url: str,
        model_id: str,
        best_of: int,
        use_beam_search: bool,
        disable_tqdm: bool,
        ssl: bool,
        ignore_eos: bool,
        stream: bool,
        cookies: Dict[str, str],
        verbose: bool,
        max_concurrent: Optional[int],
        wave: Optional[List[int]],
        logprobs: Optional[int],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        self.backend = backend
        self.api_url = api_url
        self.base_url = base_url
        self.model_id = model_id
        self.best_of = best_of
        self.use_beam_search = use_beam_search
        self.disable_tqdm = disable_tqdm
        self.ssl = ssl
        self.ignore_eos = ignore_eos
        self.stream = stream
        self.cookies = cookies
        self.verbose = verbose
        self.max_concurrent = max_concurrent
        self.wave = wave
        if wave:
            self.wave_min, self.wave_max, self.wave_sustain = wave
        self.logprobs = logprobs
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.run_id = run_id or str(uuid.uuid4())

    @property
    def request_func(
        self,
    ) -> Callable[[int, RequestFuncInput, Any, bool, float], Coroutine[Any, Any, RequestFuncOutput]]:
        return ASYNC_REQUEST_FUNCS[self.backend]

    async def send_request(
        self,
        idx: int,
        data: RequestFuncInput,
        wait_time: float,
        pbar: Optional[tqdm],
        sema: Optional[asyncio.BoundedSemaphore] = None,
    ) -> Optional[Union[RequestFuncOutput, Any]]:
        await asyncio.sleep(wait_time)
        if sema:
            async with sema:
                return await self.request_func(idx, data, pbar, self.verbose, wait_time)
        else:
            return await self.request_func(idx, data, pbar, self.verbose, wait_time)

    async def signal_profiler(
        self,
        idx: int,
        data: RequestFuncInput,
        wait_time: float,
        pbar: Optional[tqdm],
        sema: Optional[asyncio.BoundedSemaphore],
    ) -> Optional[Union[RequestFuncOutput, Any]]:
        await asyncio.sleep(wait_time)
        if sema:
            async with sema:
                return await ASYNC_REQUEST_FUNCS['profiler'](idx, data, pbar, self.verbose, wait_time)
        else:
            return await ASYNC_REQUEST_FUNCS['profiler'](idx, data, pbar, self.verbose, wait_time)

    async def send_wave_request(
        self,
        idx: int,
        data: RequestFuncInput,
        wait_time: float,
        pbar: Optional[tqdm],
        sema: asyncio.Semaphore,
        wave_state: WaveState,
    ) -> Optional[Union[RequestFuncOutput, Any]]:
        await sema.acquire()

        wave_state.num_running_requests += 1
        if wave_state.delta == 1:
            assert wave_state.num_running_requests <= self.wave_max
            if wave_state.num_running_requests == self.wave_max:
                wave_state.delta = 0
                wave_state.num_finished_requests = 0
                # If multiple requests finish before this is sent, the semaphore could have more capacity than wave_max
                # So we acquire all to sync the semaphore with our expectation
                while not sema.locked():
                    await sema.acquire()

        try:
            request_result = await self.send_request(idx, data, wait_time, pbar)
        finally:
            wave_state.num_running_requests -= 1
            wave_state.num_finished_requests += 1
            if wave_state.delta == -1:
                if wave_state.num_running_requests == self.wave_min:
                    # Don't release semaphore to keep concurrency at min
                    wave_state.delta = 0
                    wave_state.num_finished_requests = 0
                else:
                    # 50% chance of not releasing, decreasing req concurrency by 1
                    if random.getrandbits(1):
                        sema.release()
            elif wave_state.delta == 1:
                sema.release()
                # 50% chance of releasing an extra, increasing req concurrency by 1
                if random.getrandbits(1):
                    sema.release()
            else:
                sema.release()
                if wave_state.num_finished_requests == self.wave_sustain:
                    # When wave is at min, num_running_requests will be at min-1 since a request just finished
                    wave_state.delta = 1 if wave_state.num_running_requests == self.wave_min - 1 else -1

        return request_result

    async def benchmark(
        self, data: List[Tuple[str, int, int]], request_times: List[Union[float, int]], requests_media: List[List[str]]
    ) -> list[Union[RequestFuncOutput, Any, None]]:
        assert len(data) == len(request_times), "Data and request times must have the same length"
        assert len(data) == len(requests_media), "Data and request media must have the same length"
        pbar = None if self.disable_tqdm else tqdm(total=len(data))

        request_func_inputs = [
            RequestFuncInput(
                prompt=data_sample[0],
                media=media_sample,
                api_url=self.api_url,
                prompt_len=data_sample[1],
                output_len=data_sample[2],
                model=self.model_id,
                best_of=self.best_of,
                use_beam_search=self.use_beam_search,
                ssl=self.ssl,
                ignore_eos=self.ignore_eos,
                stream=self.stream,
                cookies=self.cookies,
                logprobs=self.logprobs,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                run_id=self.run_id,
            )
            for (data_sample, media_sample) in zip(data, requests_media)
        ]

        if self.wave:
            sema = asyncio.Semaphore(self.wave_max)
            wave_state = WaveState()
            return await asyncio.gather(
                *[
                    self.send_wave_request(idx, data, request_time, pbar, sema, wave_state)
                    for idx, (data, request_time) in enumerate(zip(request_func_inputs, request_times))
                ]
            )

        else:
            b_sema = asyncio.BoundedSemaphore(self.max_concurrent) if self.max_concurrent else None
            return await asyncio.gather(
                *[
                    self.send_request(idx, data, request_time, pbar, b_sema)
                    for idx, (data, request_time) in enumerate(zip(request_func_inputs, request_times))
                ]
            )

    async def validate_url_endpoint(
        self, request: Tuple[str, int, int], media_item: List[str]
    ) -> Union[RequestFuncOutput, Any]:
        data = RequestFuncInput(
            prompt=request[0],
            media=media_item,
            api_url=self.api_url,
            prompt_len=request[1],
            output_len=request[2],
            model=self.model_id,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            ssl=self.ssl,
            ignore_eos=self.ignore_eos,
            stream=self.stream,
            cookies=self.cookies,
            logprobs=self.logprobs,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        return await self.send_request(-1, data, 0, None, None)

    async def start_torch_profiler(self) -> Union[RequestFuncOutput, Any]:
        data = RequestFuncInput(
            prompt="",
            media=[],
            api_url=self.base_url + "/start_profile",
            prompt_len=0,
            output_len=0,
            model=self.model_id,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            ssl=self.ssl,
            ignore_eos=self.ignore_eos,
            stream=self.stream,
            cookies=self.cookies,
            logprobs=self.logprobs,
        )
        return await self.signal_profiler(0, data, 0, None, None)

    async def stop_torch_profiler(self) -> Union[RequestFuncOutput, Any]:
        data = RequestFuncInput(
            prompt="",
            media=[],
            api_url=self.base_url + "/stop_profile",
            prompt_len=0,
            output_len=0,
            model=self.model_id,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            ssl=self.ssl,
            ignore_eos=self.ignore_eos,
            stream=self.stream,
            cookies=self.cookies,
            logprobs=self.logprobs,
        )
        return await self.signal_profiler(0, data, 0, None, None)
