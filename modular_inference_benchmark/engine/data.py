import abc
from typing import List, Tuple
import logging
import json
import random
import transformers
from engine import distributions

logger = logging.getLogger(__name__)

PREFIX_OPTIONS = ["no-prefix", "prefix-with-text", "prefix-with-len"]


def get_data_end(
    data: List[int], tokenizer: transformers.PreTrainedTokenizer, idx: int, length: int, num_trials: int
) -> int:
    assert length >= 0 and idx >= 0
    if length == 0:
        return idx

    idy = idx + length

    def get_length(x: int, y: int) -> int:
        return len(tokenizer.encode(tokenizer.decode(data[x:y])))

    for _ in range(num_trials):
        if get_length(idx, idy) == length:
            break
        if get_length(idx, idy) < length:
            idy += 1
        else:
            idy -= 1  # Could potentially be stuck in a cycle if the length is not achievable
            if idy < idx:
                idy = idx
                break

    if get_length(idx, idy) != length:
        logger.warning(f"Tried to achieve length {length} but failed. Achieved length {get_length(idx, idy)} instead")

    return idy


class Data(abc.ABC):
    @abc.abstractmethod
    def generate_data(self, size: int) -> List[str]:
        pass


class Textfile(Data):
    def __init__(
        self,
        data: List[int],
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int,
    ) -> None:
        self.prefix_str = prefix_str
        self.prefill_distribution = prefill_distribution
        self.output_token_distribution = output_token_distribution
        self.start_distribution = distributions.AdjustedUniformInt(0, len(data) - num_trials)
        self.tokenizer = tokenizer
        self.data = data
        self.num_trials = num_trials

    @classmethod
    def with_prefix_str(
        cls,
        filename: str,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int = 10,
    ) -> "Textfile":
        with open(filename) as f:
            text = f.read()
        data = tokenizer.encode(text)

        return cls(data, prefix_str, prefill_distribution, output_token_distribution, tokenizer, num_trials)

    @classmethod
    def with_prefix_len(
        cls,
        filename: str,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int = 10,
    ) -> "Textfile":
        with open(filename) as f:
            text = f.read()
        data = tokenizer.encode(text)

        if prefix_len + num_trials >= len(data):
            raise ValueError("Prefix length is too long")

        prefix_end = get_data_end(data, tokenizer, 0, prefix_len, num_trials)  # prefix real length

        prefix_str = tokenizer.decode(data[:prefix_end]) if prefix_end > 0 else ""

        return cls(
            data[prefix_end:], prefix_str, prefill_distribution, output_token_distribution, tokenizer, num_trials
        )

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        # Can save memory by using a generator. However for performance we will use a list
        input_data = []
        lengths = self.prefill_distribution.generate_distribution(size)
        output_tokens = self.output_token_distribution.generate_distribution(size)
        starts = self.start_distribution.generate_distribution(lengths)
        prefix_len = len(self.tokenizer.encode(self.prefix_str))

        for i in range(size):
            if lengths[i] - prefix_len < 0:  # skip when sampling length less than prefix
                continue
            prompt_end = get_data_end(self.data, self.tokenizer, starts[i], lengths[i] - prefix_len, self.num_trials)
            if prompt_end < 4 or output_tokens[i] < 4:
                continue

            input_data.append(
                (
                    self.prefix_str + self.tokenizer.decode(self.data[starts[i] : prompt_end]),
                    prompt_end,
                    output_tokens[i],
                )
            )

        if len(input_data) < size:
            logger.warning(f"Could not generate the number of requests required.\nSending: {len(input_data)} requests")
            return input_data
        return random.sample(input_data, size)


class Random(Data):
    def __init__(
        self,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        token_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prefill_distribution = prefill_distribution
        self.token_distribution = token_distribution
        self.output_token_distribution = output_token_distribution
        self.prefix_str = prefix_str
        self.num_trials = num_trials

    @classmethod
    def with_prefix_str(
        cls,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int = 10,
    ) -> "Random":
        token_distribution = distributions.UniformInt(0, len(tokenizer.get_vocab()))

        return cls(
            prefix_str, prefill_distribution, token_distribution, output_token_distribution, tokenizer, num_trials
        )

    @classmethod
    def with_prefix_len(
        cls,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int = 10,
    ) -> "Random":
        token_distribution = distributions.UniformInt(0, len(tokenizer.get_vocab()))
        data = list(token_distribution.generate_distribution(prefix_len + num_trials))
        prefix_end = get_data_end(data, tokenizer, 0, prefix_len, num_trials)  # prefix real length
        prefix_str = tokenizer.decode(data[:prefix_end]) if prefix_end > 0 else ""

        return cls(
            prefix_str, prefill_distribution, token_distribution, output_token_distribution, tokenizer, num_trials
        )

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        input_data = []
        lengths = self.prefill_distribution.generate_distribution(size)
        output_tokens = self.output_token_distribution.generate_distribution(size)
        prefix_len = len(self.tokenizer.encode(self.prefix_str))

        for i in range(size):
            data = list(self.token_distribution.generate_distribution(lengths[i] + self.num_trials))
            if lengths[i] - prefix_len < 0:  # skip when sampling length less than prefix
                continue
            prompt_end = get_data_end(data, self.tokenizer, 0, lengths[i] - prefix_len, self.num_trials)
            if prompt_end < 4 or output_tokens[i] < 4:
                continue

            input_data.append(
                (self.prefix_str + self.tokenizer.decode(data[:prompt_end]), prompt_end, output_tokens[i])
            )

        if len(input_data) < size:
            logger.warning(f"Could not generate the number of requests required.\nSending: {len(input_data)} requests")
            return input_data
        return random.sample(input_data, size)


class ShareGPT(Data):
    def __init__(self, filename: str, tokenizer: transformers.PreTrainedTokenizer) -> None:
        # From https://github.com/vllm-project/vllm/blob/v0.4.0.post1/benchmarks/benchmark_serving.py#L310

        self.tokenizer = tokenizer

        with open(filename) as f:
            dataset = json.load(f)

        dataset = [data for data in dataset if len(data["conversations"]) > 2]
        tokenized_dataset = [
            (
                data["conversations"][0]["value"],
                len(tokenizer(data["conversations"][0]["value"]).input_ids),
                len(tokenizer(data["conversations"][1]["value"]).input_ids),
            )
            for data in dataset
        ]

        filtered_dataset = [
            (prompt_str, prompt_len, output_len)
            for prompt_str, prompt_len, output_len in tokenized_dataset
            if (prompt_len > 4 and output_len > 4)
        ]

        self.data = filtered_dataset

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        if len(self.data) < size:
            logger.warning(f"Could not generate the number of requests required.\nSending: {len(self.data)} requests")
            return self.data
        return random.sample(self.data, size)
