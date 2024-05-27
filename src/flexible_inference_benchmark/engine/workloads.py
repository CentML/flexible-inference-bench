import argparse
from typing import List, Union
from pydantic import BaseModel


class BaseWorkload(BaseModel):
    def overwrite_args(self, args: argparse.Namespace) -> None:
        for k, v in self.model_dump().items():
            if not getattr(args, k, None):
                continue
            setattr(args, k, v)


class General(BaseWorkload):
    request_distribution: List[Union[str, int]] = ['exponential', 2]
    input_token_distribution: List[Union[str, int]] = ['uniform', 0, 255]
    output_token_distribution: List[Union[str, int]] = ['normal', 255, 2]


class Summary(BaseWorkload):
    request_distribution: List[Union[str, int]] = ['exponential', 2]
    input_token_distribution: List[Union[str, int]] = ['uniform', 800, 1000]
    output_token_distribution: List[Union[str, int]] = ['normal', 500, 2]


class Story(BaseWorkload):
    request_distribution: List[Union[str, int]] = ['exponential', 2]
    input_token_distribution: List[Union[str, int]] = ['uniform', 200, 300]
    output_token_distribution: List[Union[str, int]] = ['normal', 1000, 20]


class Rag(BaseWorkload):
    request_distribution: List[Union[str, int]] = ['exponential', 2]
    input_token_distribution: List[Union[str, int]] = ['uniform', 400, 500]
    output_token_distribution: List[Union[str, int]] = ['normal', 1000, 5]


WORKLOADS_TYPES = {"general": General, "summary": Summary, "story": Story, "rag": Rag}
