import abc
from typing import List, Tuple
import logging
import json
import random
import transformers
from engine import distributions

logger = logging.getLogger(__name__)

PREFIX_OPTIONS = ["no-prefix","prefix-with-text","prefix-with-len"]

def get_input_text(text: str, target_token_length: int, tokenizer: transformers.PreTrainedTokenizer):
    low = 0
    high = len(text)

    while low <= high:
        mid = (low + high)//2
        curr_token_length = len(tokenizer(text[:mid]).input_ids)
        if curr_token_length == target_token_length:
            break
        elif curr_token_length < target_token_length:
            low = mid + 1
        else:
            high = mid - 1
    
    return text[:mid]

class Data(abc.ABC):
    @abc.abstractmethod
    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        pass


class Textfile(Data):
    def __init__(
        self,
        dataset_name: str,
        filename: str,
        prefix_type: str,
        prefix_text: str,
        prefix_len: int,
        prefill_distribution: distributions.Distribution,
        output_token_distribution: distributions.Distribution,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> None:
        self.prefix_type = prefix_type
        self.prefix_text = prefix_text
        self.prefix_len = prefix_len
        self.prefill_distribution = prefill_distribution
        self.output_token_distribution = output_token_distribution
        self.tokenizer = tokenizer

        if dataset_name == "other":
            with open(filename,'r') as f:
                self.text = f.read()
            self.prompt_max_tokens = len(tokenizer(self.text).input_ids)
        else: #random
            tokenizer_dict = tokenizer.get_vocab()
            self.text = " ".join(list(tokenizer_dict.keys()))
            self.prompt_max_tokens = len(tokenizer_dict)

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        factor = 1.2
        n_req = int(factor * size)
        prompt_len_arr = self.prefill_distribution.generate_distribution(n_req)
        output_token_len_arr = self.output_token_distribution.generate_distribution(n_req)
        prefix_tokens = 0
        prefix_text = ""
        if self.prefix_type == "prefix-with-text":
            prefix_tokens = len(self.tokenizer(self.prefix_text).input_ids)
            prefix_text = self.prefix_text
        elif self.prefix_type == "prefix-with-len":
            assert self.prefix_len > 0, "Prefix length cannot be negative when selecting prefix-with-len"
            prefix_tokens = min(self.prefix_len, self.prompt_max_tokens)
            prefix_text = get_input_text(self.text, prefix_tokens, self.tokenizer)

        filtered_prompts = []        
        for i in range(n_req):
            if prefix_tokens + prompt_len_arr[i] < 4 or output_token_len_arr[i] < 4:
                continue

            # make sure prompt length don't go over max tokens
            prompt_total_tokens = min(prefix_tokens + prompt_len_arr[i], self.prompt_max_tokens)
            fill_in_tokens = prompt_total_tokens - prefix_tokens
            # select a random starting point from the text to generate the remaining tokens
            start_text_idx = random.randint(0,self.prompt_max_tokens - fill_in_tokens)
            append_text = get_input_text(self.text[start_text_idx:], fill_in_tokens, self.tokenizer)
            filtered_prompts.append((prefix_text + append_text, prompt_len_arr[i], output_token_len_arr[i]))
        
        if len(filtered_prompts) < size:
            logger.warning(f"Could not generate the number of requests required.\nSending: {len(filtered_prompts)} requests")
            return filtered_prompts

        return random.sample(filtered_prompts, size)


class ShareGPT(Data):
    def __init__(self, filename: str, tokenizer: transformers.PreTrainedTokenizer) -> None:
        # From https://github.com/vllm-project/vllm/blob/v0.4.0.post1/benchmarks/benchmark_serving.py#L310

        self.tokenizer = tokenizer

        with open(filename) as f:
            dataset = json.load(f)

        dataset = [data for data in dataset if len(data["conversations"]) > 2]
        tokenized_dataset = [(
                                data["conversations"][0]["value"], 
                                len(tokenizer(data["conversations"][0]["value"]).input_ids),
                                len(tokenizer(data["conversations"][1]["value"]).input_ids)
                            ) for data in dataset]         

        filtered_dataset = [
            (prompt_str, prompt_len, output_len)
            for  prompt_str, prompt_len, output_len in tokenized_dataset
            if (
                prompt_len > 4
                and output_len > 4
            )
        ]

        self.data = filtered_dataset

    def generate_data(self, size: int) -> List[Tuple[str, int, int]]:
        if len(self.data) < size:
            logger.warning(f"Could not generate the number of requests required.\nSending: {len(self.data)} requests")
            return self.data
        return random.sample(self.data, size)
