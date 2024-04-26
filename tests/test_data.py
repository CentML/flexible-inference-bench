import random

import numpy as np
from transformers import AutoTokenizer

from engine.data import Textfile
import engine.distributions as distributions
from utils.utils import get_tokenizer

def test_random_with_prefix_str():
    random.seed(0)
    np.random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    distribution = distributions.UniformInt(10, 11)
    random_data_generator = Textfile(
        dataset_name="random",
        filename="",
        prefix_type="prefix-with-text",
        prefix_text="Hello world!", 
        prefix_len=50,
        prefill_distribution=distribution,
        output_token_distribution=distribution, 
        tokenizer=tokenizer
    )
    random_data = random_data_generator.generate_data(10)
    request = np.array(random_data)
    assert request.shape == (10,3)


def test_random_with_prefix_len():
    random.seed(0)
    np.random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
    distribution = distributions.UniformInt(10, 11)
    random_data_generator = Textfile(
        dataset_name="random",
        filename="",
        prefix_type="prefix-with-len",
        prefix_text="Hello world!", 
        prefix_len=50,
        prefill_distribution=distribution,
        output_token_distribution=distribution, 
        tokenizer=tokenizer
    )
    random_data = random_data_generator.generate_data(10)
    request = np.array(random_data)
    assert request.shape == (10,3)
