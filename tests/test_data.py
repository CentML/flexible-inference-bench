from transformers import AutoTokenizer

import modular_inference_benchmark.engine.data as data
import modular_inference_benchmark.engine.distributions as distributions


def test_random_with_prefix_str():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    distribution = distributions.UniformInt(10, 11)
    random_data_generator = data.Random.with_prefix_str(
        prefix_str="Hello world!", prefill_distribution=distribution, tokenizer=tokenizer
    )
    random_data = random_data_generator.generate_data(10)
    lens = [len(tokenizer.encode(x)) for x in random_data]
    assert lens == [10] * 10


def test_random_with_prefix_len():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    distribution = distributions.UniformInt(10, 11)
    random_data_generator = data.Random.with_prefix_len(
        prefix_len=5, prefill_distribution=distribution, tokenizer=tokenizer
    )
    random_data = random_data_generator.generate_data(10)
    lens = [len(tokenizer.encode(x)) for x in random_data]
    assert lens == [10] * 10
