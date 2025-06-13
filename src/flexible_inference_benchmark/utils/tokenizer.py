import re
import os
from dataclasses import dataclass
from transformers import AutoTokenizer  # type: ignore[attr-defined]
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from typing import Union, Any


@dataclass
class Encoding:
    input_ids: Union[list[int], list[list[int]]]


def find_tokenizer_file(files: list[str]) -> str:
    file_pattern = re.compile(r"^tokenizer\.model\.v.*$|^tekken\.json$|^tokenizer\.mm\.model\.v.*$")

    matched_files = [file for file in files if file_pattern.match(file)]
    if len(matched_files) > 1:
        raise OSError(
            f"Found {len(matched_files)} files matching the "
            f"pattern: `{file_pattern.pattern}`. Make sure there is only one tokenizer configuration"
            f"tokenizer is present in {files}."
        )
    elif len(matched_files) == 0:
        raise OSError(
            f"Found {len(matched_files)} files matching the "
            f"pattern: `{file_pattern.pattern}`. Make sure that at least there is one tokenizer configuration"
            f"tokenizer is present in {files}."
        )

    return matched_files[0]


class MistralTokenizerMode:
    def __init__(self, repo_id: str):
        assert len(repo_id.split("/")) == 2, "You have either provided a non-existent or invalid HF Hub repo id."
        repo_files = list_repo_files(repo_id=repo_id, repo_type="model")
        filename = find_tokenizer_file(repo_files)
        file = hf_hub_download(repo_id=repo_id, filename=filename, token=os.environ.get("HF_TOKEN", ""))
        tokenizer = MistralTokenizer.from_file(file)

        tokenizer_ = tokenizer.instruct_tokenizer.tokenizer
        from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy, Tekkenizer

        self.is_tekken = isinstance(tokenizer_, Tekkenizer)
        from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer

        self.is_spm = isinstance(tokenizer_, SentencePieceTokenizer)
        if self.is_tekken:
            # Make sure special tokens will not raise
            tokenizer_.special_token_policy = SpecialTokenPolicy.IGNORE  # type: ignore[attr-defined]
        elif self.is_spm:
            pass
        else:
            raise TypeError(f"Unsupported tokenizer: {type(tokenizer_)}")

        self._vocab = tokenizer_.vocab()
        self._vocab_dict = {token: idx for idx, token in enumerate(self._vocab)}
        self.tokenizer = tokenizer_
        self.name_or_path = repo_id

    def __call__(self, text: Union[str, list[str]]) -> Encoding:
        input_ids: Union[list[int], list[list[int]]]
        if isinstance(text, str):
            input_ids = self.encode(text)
            return Encoding(input_ids=input_ids)
        elif isinstance(text, list):
            input_ids = [self.encode(t) for t in text]
            return Encoding(input_ids=input_ids)
        else:
            raise TypeError(f"Unsupported type for text: {type(text)}. Expected str or list[str].")

    def get_vocab(self) -> dict[str, int]:
        return self._vocab_dict

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, bos=True, eos=False)

    def decode(self, ids: Union[list[int], int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens, "skip_special_tokens=False is not supported for Mistral tokenizers."
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.decode(ids)


def select_tokenizer(tokenizer_id: str, tokenizer_mode: str) -> Any:
    if tokenizer_mode == "mistral":
        return MistralTokenizerMode(tokenizer_id)

    return AutoTokenizer.from_pretrained(tokenizer_id)
