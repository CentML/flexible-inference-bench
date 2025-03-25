import abc
from typing import List, Tuple, Optional
import logging
import json
import random
import sqlite3
import os
from hashlib import sha256

import transformers
from flexible_inference_benchmark.engine import distributions
from flexible_inference_benchmark.engine.backend_functions import (
    RequestPromptData,
)
logger = logging.getLogger(__name__)


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
        logger.debug(f"Tried to achieve length {length} but failed. Achieved length {get_length(idx, idy)} instead")

    return idy


def hash_string(s: str) -> str:
    return sha256(s.encode()).hexdigest()


class Data(abc.ABC):
    @abc.abstractmethod
    def generate_data(self, size: int) -> List[RequestPromptData]:
        pass


class Textfile(Data):
    def __init__(
        self,
        data: List[int],
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int,
    ) -> None:
        self.prefix_str = prefix_str
        self.prefill_distribution = prefill_distribution
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
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int = 10,
    ) -> "Textfile":
        with open(filename) as f:
            text = f.read()
        data = tokenizer.encode(text)

        return cls(data, prefix_str, prefill_distribution, tokenizer, num_trials)

    @classmethod
    def with_prefix_len(
        cls,
        filename: str,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
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
            data[prefix_end:], prefix_str, prefill_distribution, tokenizer, num_trials
        )

    def generate_data(self, size: int) -> List[RequestPromptData]:
        # Can save memory by using a generator. However for performance we will use a list
        input_data = []
        lengths = self.prefill_distribution.generate_distribution(size)
        starts = self.start_distribution.generate_distribution(lengths)
        prefix_len = len(self.tokenizer.encode(self.prefix_str))

        for i in range(size):
            if lengths[i] - prefix_len < 0:  # skip when sampling length less than prefix
                continue
            prompt_end = get_data_end(self.data, self.tokenizer, starts[i], lengths[i] - prefix_len, self.num_trials)

            input_data.append(RequestPromptData.build(
                tokenizer=self.tokenizer,
                completion_prompt_str=self.prefix_str + self.tokenizer.decode(self.data[starts[i] : prompt_end]),
                completion_prompt_num_tokens=prefix_len + (prompt_end - starts[i])))

        if len(input_data) < size:
            logger.warning(f"Generating {len(input_data)} requests instead of {size} requests.")
            return input_data
        return random.sample(input_data, size)


class Random(Data):
    def __init__(
        self,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        token_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prefill_distribution = prefill_distribution
        self.token_distribution = token_distribution
        self.prefix_str = prefix_str
        self.num_trials = num_trials

    @classmethod
    def with_prefix_str(
        cls,
        prefix_str: str,
        prefill_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int = 10,
    ) -> "Random":
        token_distribution = distributions.UniformInt(0, len(tokenizer.get_vocab()))

        return cls(
            prefix_str, prefill_distribution, token_distribution, tokenizer, num_trials
        )

    @classmethod
    def with_prefix_len(
        cls,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
        num_trials: int = 10,
    ) -> "Random":
        token_distribution = distributions.UniformInt(0, len(tokenizer.get_vocab()))
        data = list(token_distribution.generate_distribution(prefix_len + num_trials))
        prefix_end = get_data_end(data, tokenizer, 0, prefix_len, num_trials)  # prefix real length
        prefix_str = tokenizer.decode(data[:prefix_end]) if prefix_end > 0 else ""

        return cls(
            prefix_str, prefill_distribution, token_distribution, tokenizer, num_trials
        )

    def generate_data(self, size: int) -> List[RequestPromptData]:
        input_data = []
        lengths = self.prefill_distribution.generate_distribution(size)
        prefix_len = len(self.tokenizer.encode(self.prefix_str))

        for i in range(size):
            data = list(self.token_distribution.generate_distribution(lengths[i] + self.num_trials))
            if lengths[i] - prefix_len < 0:  # skip when sampling length less than prefix
                continue
            prompt_end = get_data_end(data, self.tokenizer, 0, lengths[i] - prefix_len, self.num_trials)

            input_data.append(RequestPromptData.build(
                tokenizer=self.tokenizer,
                completion_prompt_str=self.prefix_str + self.tokenizer.decode(data[:prompt_end]),
                completion_prompt_num_tokens=prefix_len + prompt_end))

        if len(input_data) < size:
            logger.warning(f"Generating {len(input_data)} requests instead of {size} requests.")
            return input_data
        return random.sample(input_data, size)


class ShareGPT(Data):
    def __init__(self, filename: str, tokenizer: transformers.PreTrainedTokenizer) -> None:
        # From https://github.com/vllm-project/vllm/blob/v0.4.0.post1/benchmarks/benchmark_serving.py#L310

        self.tokenizer = tokenizer

        with open(filename) as f:
            dataset = json.load(f)

        dataset = [data for data in dataset if len(data["conversations"]) >= 2]

        os.makedirs(os.path.expanduser("~/.cache/flexible_inference_benchmark/"), exist_ok=True)

        database_cache_path = os.path.join(
            os.path.expanduser("~/.cache/flexible_inference_benchmark/"), "sharegpt_database_v0.db"
        )

        # create the database if it does not exist
        if not os.path.exists(database_cache_path):
            with open(database_cache_path, "w"):
                connection = sqlite3.connect(database_cache_path)
                cursor = connection.cursor()
                cursor.execute(
                    "CREATE TABLE sharegpt (index INTEGER PRIMARY KEY, prompt TEXT, output TEXT)"
                )
                insert_data = [(i, data["conversations"][0]["value"], data["conversations"][1]["value"]) for i, data in enumerate(dataset)]
                insert_data = list(filter(lambda x: len(tokenizer.encode(x[1])) > 4 and len(tokenizer.encode(x[2])) > 4, insert_data))
                cursor.executemany(
                    "INSERT INTO sharegpt (index, prompt, output) VALUES (?, ?, ?)", insert_data
                )
                cursor.execute(
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)"
                )
                cursor.execute(
                    "INSERT INTO metadata (key, value) VALUES ('version', '0')"
                )
                cursor.execute(
                    "INSERT INTO metadata (key, value) VALUES ('count', ?)", (str(len(insert_data)),)
                )
                connection.commit()
                connection.close()

        self.database_cache_path = database_cache_path
        return

        tokenizer_id = tokenizer.name_or_path.replace("/", "_")
        cache_path = os.path.join(
            os.path.expanduser("~/.cache/flexible_inference_benchmark/"), f"sharegpt_sizes_{tokenizer_id}.json"
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

        # TODO: fix this
        # TODO: make sure output distribution gets calculated
        filtered_dataset = [
            RequestPromptData.build(
                tokenizer,
                completion_prompt_str=prompt_str,

            )
            RequestPromptData.from_completion_string(prompt_str, tokenizer)
            for prompt_str, prompt_len, output_len in tokenized_dataset
            if (prompt_len > 4 and output_len > 4)
        ]

        self.data = filtered_dataset

        logger.info("Loaded ShareGPT dataset.")

    def generate_data(self, size: int) -> List[RequestPromptData]:
        # establish a connection to the database
        connection = sqlite3.connect(self.database_cache_path)
        cursor = connection.cursor()

        # read the number of entries from the metadata table
        cursor.execute("SELECT value FROM metadata WHERE key='count'")
        count = int(cursor.fetchone()[0])

        sample_num = min(size, count)
        indices = random.sample(range(count), sample_num)

        # read the entries from the database
        cursor.execute("SELECT prompt, output FROM sharegpt WHERE index IN ({placeholder})", tuple(indices))
        data = cursor.fetchall()

        # close the connection
        connection.close()

        # format the request data
        data = [
            RequestPromptData.build(
                tokenizer=self.tokenizer,
                completion_prompt_str=prompt,
                expected_completion=output
            )
            for prompt, output in data
        ]

        if sample_num < size:
            logger.warning(f"Generating {count} requests instead of {size} requests.")
        return data

# fib benchmark -n 10 -rps 2 --dataset json --disable-ignore-eos --backend openai-chat --endpoint v1/chat/completions
class JSONModeEval(Data):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        from datasets import load_dataset
        import json
        ds = load_dataset("NousResearch/json-mode-eval")["train"]
        data_list = []
        for row in ds:
            messages = row["prompt"]
            schema = json.loads(row["schema"])
            data_list.append(RequestPromptData.build(
                tokenizer=tokenizer,
                chat_prompt_messages=messages,
                expected_completion=row["completion"],
                schema=schema
            ))

        self.data = data_list
        logger.info("Loaded JSON Mode Eval dataset.")

    def generate_data(self, size: int) -> List[RequestPromptData]:
        if len(self.data) < size:
            logger.warning(f"Generating {len(self.data)} requests instead of {size} requests.")
            return self.data
        return random.sample(self.data, size)
