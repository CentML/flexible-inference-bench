from transformers import AutoTokenizer

import modular_inference_benchmark.engine.data as data
import modular_inference_benchmark.engine.distributions as distributions


def test_random_with_prefix_str():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    distribution = distributions.UniformInt(10, 20)
    random_data_generator = data.Random.with_prefix_str(
        prefix_str="Hello, world!", prefill_distribution=distribution, tokenizer=tokenizer
    )
    random_data = random_data_generator.generate_data(10)
    assert len(random_data) == 10


def test_random_with_prefix_len():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    distribution = distributions.UniformInt(10, 20)
    random_data_generator = data.Random.with_prefix_len(
        prefix_len=10, prefill_distribution=distribution, tokenizer=tokenizer
    )
    random_data = random_data_generator.generate_data(10)
    assert len(random_data) == 10
