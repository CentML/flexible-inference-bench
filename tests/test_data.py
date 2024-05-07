import random
import os 
import numpy as np
from transformers import AutoTokenizer
import json
import modular_inference_benchmark.engine.data as data
import modular_inference_benchmark.engine.distributions as distributions
from sharegpt_data import SHAREGPT_DATA

def test_random_with_prefix_str():
    random.seed(0)
    np.random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    distribution = distributions.UniformInt(10, 11)
    output_token_distribution = distributions.UniformInt(10,11)
    random_data_generator = data.Random.with_prefix_str(
        prefix_str="Hello world!", prefill_distribution=distribution, output_token_distribution=output_token_distribution, tokenizer=tokenizer
    )
    random_data = np.array(random_data_generator.generate_data(10))
    assert random_data.shape == (10,3)


def test_random_with_prefix_len():
    random.seed(0)
    np.random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    distribution = distributions.UniformInt(10, 11)
    output_token_distribution = distributions.UniformInt(10, 11)
    random_data_generator = data.Random.with_prefix_len(
        prefix_len=5, prefill_distribution=distribution, output_token_distribution=output_token_distribution, tokenizer=tokenizer
    )
    random_data = np.array(random_data_generator.generate_data(10))
    assert random_data.shape == (10,3)

def test_sharegpt():
    with open('sharegpt_test.json', 'w') as f:
        json.dump(SHAREGPT_DATA, f)
    filename = 'sharegpt_test.json'
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sharegptClass = data.ShareGPT(filename, tokenizer)
    random_data = np.array(sharegptClass.generate_data(10))
    if os.path.exists("sharegpt_test.json"):
        os.remove("sharegpt_test.json")
    assert random_data.shape == (10,3)
