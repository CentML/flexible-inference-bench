import asyncio
import logging
from typing import List, Tuple, Callable, Iterable, Optional
from tqdm import tqdm
from modular_inference_benchmark.engine.backend_functions import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)

logger = logging.getLogger(__name__)


class Client:
    def __init__(
        self, backend: str, api_url: str, model_id: str, best_of: int, use_beam_search: bool, disable_tqdm: bool
    ):
        self.backend = backend
        self.api_url = api_url
        self.model_id = model_id
        self.best_of = best_of
        self.use_beam_search = use_beam_search
        self.disable_tqdm = disable_tqdm

    @property
    def request_func(self) -> Callable:  # type: ignore
        return ASYNC_REQUEST_FUNCS[self.backend]

    async def send_request(self, data: RequestFuncInput, wait_time: float, pbar: Optional[tqdm]) -> RequestFuncOutput:
        logger.debug(f"Sending request with data {data} and wait time {wait_time}")
        await asyncio.sleep(wait_time)
        return await self.request_func(data, pbar)  # type: ignore

    async def benchmark(
        self, data: List[Tuple[str, int, int]], request_times: Iterable  # type: ignore
    ) -> RequestFuncOutput:
        assert len(data) == len(request_times), "Data and request times must have the same length"  # type: ignore
        pbar = None if self.disable_tqdm else tqdm(total=len(data))

        request_func_inputs = [
            RequestFuncInput(
                data_sample[0],
                self.api_url,
                data_sample[1],
                data_sample[2],
                self.model_id,
                self.best_of,
                self.use_beam_search,
            )
            for data_sample in data
        ]

        return await asyncio.gather(  # type: ignore
            *[
                self.send_request(data, request_time, pbar)
                for data, request_time in zip(request_func_inputs, request_times)
            ]
        )
