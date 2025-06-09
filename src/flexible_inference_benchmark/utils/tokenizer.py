import re
import os
from transformers import AutoTokenizer  # type: ignore[attr-defined]
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # type: ignore[attr-defined]
from huggingface_hub import hf_hub_download 
from huggingface_hub import list_repo_files
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

def find_tokenizer_file(files: list[str]):
    file_pattern = re.compile(
        r"^tokenizer\.model\.v.*$|^tekken\.json$|^tokenizer\.mm\.model\.v.*$")

    matched_files = [file for file in files if file_pattern.match(file)]
    if len(matched_files) > 1:
        raise OSError(
            f"Found {len(matched_files)} files matching the "
            f"pattern: `{file_pattern.pattern}`. Make sure there is only one tokenizer configuration"
            f"tokenizer is present in {files}.")
    elif len(matched_files) == 0:
        raise OSError(
            f"Found {len(matched_files)} files matching the "
            f"pattern: `{file_pattern.pattern}`. Make sure that at least there is one tokenizer configuration"
            f"tokenizer is present in {files}.")

    return matched_files[0]

class MistralTokenizerMode(PreTrainedTokenizerBase):
  def __init__(self, repo_id: str):
    assert len(repo_id.split("/")) == 2, (
        "You have either provided a non-existent or invalid HF Hub repo id.")
    repo_files = list_repo_files(repo_id=repo_id, repo_type="model")
    filename = find_tokenizer_file(repo_files)
    file = hf_hub_download(repo_id=repo_id, filename=filename, token=os.environ.get("HF_TOKEN",""))
    tokenizer = MistralTokenizer.from_file(file)

    tokenizer_ = tokenizer.instruct_tokenizer.tokenizer
    from mistral_common.tokens.tokenizers.tekken import (
        SpecialTokenPolicy, Tekkenizer)
    self.is_tekken = isinstance(tokenizer_, Tekkenizer)
    from mistral_common.tokens.tokenizers.sentencepiece import (
        SentencePieceTokenizer)
    self.is_spm = isinstance(tokenizer_, SentencePieceTokenizer)
    if self.is_tekken:
        # Make sure special tokens will not raise
        tokenizer_.special_token_policy = SpecialTokenPolicy.IGNORE
    elif self.is_spm:
        pass
    else:
        raise TypeError(f"Unsupported tokenizer: {type(tokenizer_)}")

    self._vocab = tokenizer_.vocab()
    self.tokenizer = tokenizer_

  def get_vocab(self):
     return self._vocab
  
  def encode(self, text:str):
     return self.tokenizer.encode(text, bos=None, eos=None)
  
  def decode(self, ids: list[int]):
     return self.tokenizer.decode(ids)
    
def select_tokenizer(tokenizer_id: str, tokenizer_mode: str) -> PreTrainedTokenizerBase:
  if not tokenizer_mode:
    return AutoTokenizer.from_pretrained(tokenizer_id)
  if tokenizer_mode == "mistral":
    return MistralTokenizerMode(tokenizer_id)