import asyncio
import logging
from typing import List, Tuple, Callable, Optional, Any, Coroutine, Union, Dict
from tqdm import tqdm
from flexible_inference_benchmark.engine.backend_functions import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)

logger = logging.getLogger(__name__)


class Client:
    def __init__(
        self,
        backend: str,
        api_url: str,
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
    ):
        self.backend = backend
        self.api_url = api_url
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
        sema: Optional[asyncio.BoundedSemaphore],
    ) -> Optional[Union[RequestFuncOutput, Any]]:
        await asyncio.sleep(wait_time)
        if sema:
            async with sema:
                return await self.request_func(idx, data, pbar, self.verbose, wait_time)
        else:
            return await self.request_func(idx, data, pbar, self.verbose, wait_time)

    async def benchmark(
        self, data: List[Tuple[str, int, int]], request_times: List[Union[float, int]]
    ) -> list[Union[RequestFuncOutput, Any, None]]:
        assert len(data) == len(request_times), "Data and request times must have the same length"
        pbar = None if self.disable_tqdm else tqdm(total=len(data))

        request_func_inputs = [
            RequestFuncInput(
                prompt=data_sample[0],
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
            )
            for data_sample in data
        ]

        sema = asyncio.BoundedSemaphore(self.max_concurrent) if self.max_concurrent else None

        return await asyncio.gather(
            *[
                self.send_request(idx, data, request_time, pbar, sema)
                for idx, (data, request_time) in enumerate(zip(request_func_inputs, request_times))
            ]
        )

    async def validate_url_endpoint(self, request: Tuple[str, int, int]) -> Union[RequestFuncOutput, Any]:
        data = RequestFuncInput(
            prompt=request[0],
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
        )
        return await self.send_request(0, data, 0, None, None)
