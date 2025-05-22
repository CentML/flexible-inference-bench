# pylint: disable=too-many-positional-arguments
import abc
from typing import List, Tuple, Optional
import logging
import json
import random
import os
import tempfile
import shutil
import numpy as np
from hashlib import sha256

import librosa
import soundfile
from datasets import load_dataset  # type: ignore[attr-defined]
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


class ASRDataset(Data):
    """
    Dataset class for loading and preparing audio data for ASR benchmarking.
    Originally inspired from vLLM's ASR dataset class.
    
    This class loads audio samples from a Hugging Face dataset, prepares them for
    transcription benchmarking, and manages temporary storage of audio files.
    """
    DEFAULT_AUDIO_PREAMBLE_TEMPLATE = "<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>"
    TEXT_FIELD_CANDIDATES = ['text', 'transcription', 'sentence']

    def __init__(
        self,
        hf_dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        hf_dataset_config: Optional[str] = None,
        hf_dataset_split: str = "train",
        language: str = "en",
        preamble_template: Optional[str] = None,
        audio_duration_limit_sec: Optional[float] = 30.0,
        audio_column: str = "audio",
        text_column: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.hf_dataset_name = hf_dataset_name
        self.hf_dataset_config = hf_dataset_config
        self.hf_dataset_split = hf_dataset_split
        self.language = language
        self.preamble = (preamble_template or self.DEFAULT_AUDIO_PREAMBLE_TEMPLATE).format(lang=language)
        self.audio_duration_limit_sec = audio_duration_limit_sec
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_samples = max_samples

        self.temp_dir = tempfile.mkdtemp(prefix="fib_audio_cache_")
        logger.info(f"Created temporary directory for audio files: {self.temp_dir}")

        self.dataset_samples: List[Tuple[str, int, int, str]] = self._load_and_prepare_data()

    def _find_text_column(self, features) -> str:
        """Find the column containing the transcription text."""
        if self.text_column and self.text_column in features:
            return self.text_column
        for candidate in self.TEXT_FIELD_CANDIDATES:
            if candidate in features:
                logger.info(f"Using '{candidate}' as text column for ASR dataset.")
                return candidate
        raise ValueError(f"Could not find a suitable text column (tried: {self.TEXT_FIELD_CANDIDATES}) in dataset features: {list(features.keys())}")

    def _load_and_prepare_data(self) -> List[Tuple[str, int, int, str]]:
        """Load and prepare audio data from a Hugging Face dataset."""
        try:
            dataset = load_dataset(
                self.hf_dataset_name,
                name=self.hf_dataset_config,
                split=self.hf_dataset_split,
                streaming=False,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {self.hf_dataset_name}: {e}")
            return []

        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        prepared_data = []
        actual_text_column = self._find_text_column(dataset.features)

        logger.info(f"Preparing ASR data from {self.hf_dataset_name} ({self.hf_dataset_split}). This may take a while...")
        
        processed_count = 0
        skipped_duration = 0
        skipped_missing_text = 0

        for i, item in enumerate(dataset):
            if self.audio_column not in item or not item[self.audio_column]:
                logger.warning(f"Skipping item {i} due to missing or empty audio data in column '{self.audio_column}'.")
                continue
            
            audio_data = item[self.audio_column]
            
            if not isinstance(audio_data, dict) or "array" not in audio_data or "sampling_rate" not in audio_data:
                logger.warning(f"Skipping item {i} due to unexpected audio data format: {type(audio_data)}. Expected dict with 'array' and 'sampling_rate'.")
                continue

            y = np.array(audio_data["array"])
            sr = audio_data["sampling_rate"]

            if y.ndim > 1: # If stereo, convert to mono.
                y = librosa.to_mono(y.T)

            duration_s = librosa.get_duration(y=y, sr=sr)
            if self.audio_duration_limit_sec and duration_s > self.audio_duration_limit_sec:
                skipped_duration += 1
                continue

            reference_text = item.get(actual_text_column)
            if not reference_text or not isinstance(reference_text, str):
                skipped_missing_text +=1
                continue
            
            # Save audio to a temporary WAV file. Using a unique name based on index to avoid collisions if multiple identical audios exist.
            temp_audio_filename = os.path.join(self.temp_dir, f"audio_sample_{processed_count}.wav")
            try:
                soundfile.write(temp_audio_filename, y, sr, format="WAV")
            except Exception as e:
                logger.error(f"Failed to write temporary audio file for item {i}: {e}")
                continue

            prompt_len = len(self.tokenizer.encode(self.preamble))
            output_len = len(self.tokenizer.encode(reference_text)) # Expected output tokens

            prepared_data.append((self.preamble, prompt_len, output_len, temp_audio_filename))
            processed_count += 1
        
        if skipped_duration > 0:
            logger.info(f"Skipped {skipped_duration} audio samples due to duration limit ({self.audio_duration_limit_sec}s).")
        if skipped_missing_text > 0:
            logger.info(f"Skipped {skipped_missing_text} audio samples due to missing reference text.")
        
        logger.info(f"Successfully prepared {len(prepared_data)} ASR samples.")
        return prepared_data

    def generate_data(self, size: int) -> List[Tuple[str, int, int, str]]:
        """
        Generate random samples from the prepared dataset.
        
        Returns a list of tuples, where each tuple contains:
        (preamble_text, prompt_len, output_len, audio_file_path)
        """
        if not self.dataset_samples:
            logger.warning("ASR dataset is empty or failed to load. Returning no data.")
            return []
        
        if len(self.dataset_samples) < size:
            logger.warning(
                f"Requested {size} samples, but ASR dataset only has {len(self.dataset_samples)}. "
                f"Returning all available samples. Consider increasing --max-samples or using a larger dataset split."
            )
            return self.dataset_samples
        return random.sample(self.dataset_samples, size)

    def cleanup_temp_dir(self):
        """Explicitly clean up temporary files."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Successfully cleaned up temporary audio directory: {self.temp_dir}")
                self.temp_dir = None
            except OSError as e:
                logger.error(f"Error cleaning up temporary directory {self.temp_dir}: {e}")
    
    def __del__(self):
        """Clean up temporary files on object destruction as a fallback."""
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            logger.warning(
                f"ASRDataset temporary directory {self.temp_dir} was not explicitly cleaned. "
                "Attempting cleanup in __del__. Please ensure cleanup_temp_dir() is called."
            )
            self.cleanup_temp_dir()


class ShareGPT(Data):
    def __init__(self, filename: str, tokenizer: PreTrainedTokenizerBase) -> None:
        # From https://github.com/vllm-project/vllm/blob/v0.4.0.post1/benchmarks/benchmark_serving.py#L310

        self.tokenizer = tokenizer

        with open(filename) as f:
            dataset = json.load(f)

        dataset = [data for data in dataset if len(data["conversations"]) >= 2]

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

        filtered_dataset = [
            (prompt_str, prompt_len, output_len)
            for prompt_str, prompt_len, output_len in tokenized_dataset
            if (prompt_len > 4 and output_len > 4)
        ]

        self.data = filtered_dataset

        logger.info("Loaded ShareGPT dataset.")

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        if len(self.data) < size:
            logger.debug(f"Generating {len(self.data)} requests instead of {size} requests.")
            return self.data
        return random.sample(self.data, size)
