import argparse
from typing import List, Union
from pydantic import BaseModel


class SnowflakeWorkload(BaseModel):
    request_distribution: List[Union[str, int]] = ['exponential', 2]
    input_token_distribution: List[Union[str, int]] = ['uniform', 0, 255]
    output_token_distribution: List[Union[str, int]] = ['normal', 255, 2]

    def overwrite_args(self: BaseModel, args: argparse.Namespace) -> None:
        for k, v in self.model_dump().items():
            if not getattr(args, k, None):
                continue
            setattr(args, k, v)


WORKLOADS_TYPES = {"snowflake": SnowflakeWorkload}
