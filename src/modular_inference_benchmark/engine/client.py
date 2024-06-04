import asyncio
import logging
from typing import List, Tuple, Callable, Optional, Any, Coroutine, Union
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
    ):
        self.backend = backend
        self.api_url = api_url
        self.model_id = model_id
        self.best_of = best_of
        self.use_beam_search = use_beam_search
        self.disable_tqdm = disable_tqdm
        self.ssl = ssl
        self.ignore_eos = ignore_eos

    @property
    def request_func(self) -> Callable[[RequestFuncInput, Any | None], Coroutine[Any, Any, RequestFuncOutput]]:
        return ASYNC_REQUEST_FUNCS[self.backend]

    async def send_request(
        self, data: RequestFuncInput, wait_time: float, pbar: Optional[tqdm]
    ) -> Optional[RequestFuncOutput | Any]:
        logger.debug(f"Sending request with data {data} and wait time {wait_time}")
        await asyncio.sleep(wait_time)
        return await self.request_func(data, pbar)

    async def benchmark(
        self, data: List[Tuple[str, int, int]], request_times: List[float | int]
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
            )
            for data_sample in data
        ]

        return await asyncio.gather(
            *[
                self.send_request(data, request_time, pbar)
                for data, request_time in zip(request_func_inputs, request_times)
            ]
        )
