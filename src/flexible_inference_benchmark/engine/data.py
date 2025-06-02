# pylint: disable=too-many-positional-arguments
import abc
from typing import List, Tuple, Optional
import logging
import json
import random
import os
from hashlib import sha256

from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # type: ignore[attr-defined]
from flexible_inference_benchmark.engine import distributions

logger = logging.getLogger(__name__)


def get_data_end(data: List[int], tokenizer: PreTrainedTokenizerBase, idx: int, length: int, num_trials: int) -> int:
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
        logger.debug(f"Tried to achieve length {length} but failed. Achieved length {get_length(idx, idy)} instead")

    return idy


def hash_string(s: str) -> str:
    return sha256(s.encode()).hexdigest()


class Data(abc.ABC):
    @abc.abstractmethod
    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        pass


class Textfile(Data):
    def __init__(
        self,
        data: List[int],
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        num_trials: int,
        ignore_input_distribution: bool,
    ) -> None:
        self.prefix_str = prefix_str
        self.prefill_distribution = prefill_distribution
        self.output_token_distribution = output_token_distribution
        self.start_distribution = distributions.AdjustedUniformInt(0, len(data) - num_trials)
        self.tokenizer = tokenizer
        self.data = data
        self.num_trials = num_trials
        self.ignore_input_distribution = ignore_input_distribution

    @classmethod
    def with_prefix_str(
        cls,
        filename: str,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        ignore_input_distribution: bool,
        num_trials: int = 10,
    ) -> "Textfile":
        with open(filename) as f:
            text = f.read()
        data = tokenizer.encode(text)

        return cls(
            data,
            prefix_str,
            prefill_distribution,
            output_token_distribution,
            tokenizer,
            num_trials,
            ignore_input_distribution,
        )

    @classmethod
    def with_prefix_len(
        cls,
        filename: str,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        ignore_input_distribution: bool,
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
            data[prefix_end:],
            prefix_str,
            prefill_distribution,
            output_token_distribution,
            tokenizer,
            num_trials,
            ignore_input_distribution,
        )

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        # Can save memory by using a generator. However for performance we will use a list
        input_data: List[Tuple[str, int, int]] = []
        lengths = self.prefill_distribution.generate_distribution(size)
        output_tokens = self.output_token_distribution.generate_distribution(size)
        starts = self.start_distribution.generate_distribution(lengths)
        prefix_len = len(self.tokenizer.encode(self.prefix_str))

        if self.ignore_input_distribution:
            input_data = [(self.prefix_str, prefix_len, output_tokens[i]) for i in range(size)]
        else:
            for i in range(size):
                if lengths[i] - prefix_len < 0:  # skip when sampling length less than prefix
                    continue
                prompt_end = get_data_end(
                    self.data, self.tokenizer, starts[i], lengths[i] - prefix_len, self.num_trials
                )
                achieved_len = (prompt_end - starts[i]) + prefix_len

                input_data.append(
                    (
                        self.prefix_str + self.tokenizer.decode(self.data[starts[i] : prompt_end]),
                        achieved_len,
                        output_tokens[i],
                    )
                )

        if len(input_data) < size:
            logger.debug(f"Generating {len(input_data)} requests instead of {size} requests.")
            return input_data

        return random.sample(input_data, size)


class Random(Data):
    def __init__(
        self,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        token_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        num_trials: int,
        ignore_input_distribution: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.prefill_distribution = prefill_distribution
        self.token_distribution = token_distribution
        self.output_token_distribution = output_token_distribution
        self.prefix_str = prefix_str
        self.num_trials = num_trials
        self.ignore_input_distribution = ignore_input_distribution

    @classmethod
    def with_prefix_str(
        cls,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        ignore_input_distribution: bool,
        num_trials: int = 10,
    ) -> "Random":
        ## Specifying the middle 50% range to avoid accidental generation of <image> tokens
        token_distribution = distributions.UniformInt(
            len(tokenizer.get_vocab()) // 4, 3 * len(tokenizer.get_vocab()) // 4
        )

        return cls(
            prefix_str,
            prefill_distribution,
            token_distribution,
            output_token_distribution,
            tokenizer,
            num_trials,
            ignore_input_distribution,
        )

    @classmethod
    def with_prefix_len(
        cls,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: PreTrainedTokenizerBase,
        ignore_input_distribution: bool,
        num_trials: int = 10,
    ) -> "Random":
        token_distribution = distributions.UniformInt(0, len(tokenizer.get_vocab()))
        data = list(token_distribution.generate_distribution(prefix_len + num_trials))
        prefix_end = get_data_end(data, tokenizer, 0, prefix_len, num_trials)  # prefix real length
        prefix_str = tokenizer.decode(data[:prefix_end]) if prefix_end > 0 else ""

        return cls(
            prefix_str,
            prefill_distribution,
            token_distribution,
            output_token_distribution,
            tokenizer,
            num_trials,
            ignore_input_distribution,
        )

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        input_data: List[Tuple[str, int, int]] = []
        lengths = self.prefill_distribution.generate_distribution(size)
        output_tokens = self.output_token_distribution.generate_distribution(size)
        prefix_len = len(self.tokenizer.encode(self.prefix_str))

        if self.ignore_input_distribution:
            input_data = [(self.prefix_str, prefix_len, output_tokens[i]) for i in range(size)]
        else:
            for i in range(size):
                data = list(self.token_distribution.generate_distribution(lengths[i] + self.num_trials))
                if lengths[i] - prefix_len < 0:  # skip when sampling length less than prefix
                    continue
                prompt_end = get_data_end(data, self.tokenizer, 0, lengths[i] - prefix_len, self.num_trials)
                achieved_len = prompt_end + prefix_len

                input_data.append(
                    (self.prefix_str + self.tokenizer.decode(data[:prompt_end]), achieved_len, output_tokens[i])
                )

        if len(input_data) < size:
            logger.debug(f"Generating {len(input_data)} requests instead of {size} requests.")
            return input_data
        return random.sample(input_data, size)


class ShareGPT(Data):
    def __init__(
        self,
        filename: str,
        tokenizer: PreTrainedTokenizerBase,
        output_token_distribution: Optional[distributions.Distribution] = None,
    ) -> None:
        # From https://github.com/vllm-project/vllm/blob/v0.4.0.post1/benchmarks/benchmark_serving.py#L310

        self.tokenizer = tokenizer

        with open(filename) as f:
            dataset = json.load(f)

        dataset = [data for data in dataset if len(data["conversations"]) >= 2]

        tokenizer_id = tokenizer.name_or_path.replace("/", "_")
        os.makedirs(os.path.expanduser("~/.cache/flexible_inference_benchmark/"), exist_ok=True)
        cache_path = os.path.join(
            os.path.expanduser("~/.cache/flexible_inference_benchmark/"),
            f"sharegpt_sizes_{filename.replace('.', '-').replace('/', '-')}_{tokenizer_id}.json",
        )
        try:
            with open(cache_path, "r") as fcache:
                length_cache = json.load(fcache)
        except (FileNotFoundError, json.JSONDecodeError):
            length_cache = {}

        sequences_to_encode = [data["conversations"][0]["value"] for data in dataset] + [
            data["conversations"][1]["value"] for data in dataset
        ]
        all_in_cache = len(length_cache) > 0 and all(hash_string(seq) in length_cache for seq in sequences_to_encode)
        if not all_in_cache:
            encoded = tokenizer(sequences_to_encode)
            for i, seq in enumerate(sequences_to_encode):
                length_cache[hash_string(seq)] = len(encoded.input_ids[i])
            with open(cache_path, "w") as fcache:
                json.dump(length_cache, fcache)
        results_input_ids = [length_cache[hash_string(seq)] for seq in sequences_to_encode]
        tokenized_dataset = [
            (dataset[i]["conversations"][0]["value"], results_input_ids[i], results_input_ids[i + len(dataset)])
            for i in range(len(dataset))
        ]

        if output_token_distribution is not None:
            output_lengths = output_token_distribution.generate_distribution(len(tokenized_dataset))
            for i in range(len(tokenized_dataset)):
                tokenized_dataset[i] = (tokenized_dataset[i][0], tokenized_dataset[i][1], output_lengths[i])

        filtered_dataset = [
            (prompt_str, prompt_len, output_len)
            for prompt_str, prompt_len, output_len in tokenized_dataset
            if (prompt_len > 4 and output_len > 4)
        ]

        self.data = filtered_dataset

        logger.info("Loaded ShareGPT-formatted dataset.")

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        if len(self.data) < size:
            logger.debug(f"Generating {len(self.data)} requests instead of {size} requests.")
            return self.data
        return random.sample(self.data, size)
