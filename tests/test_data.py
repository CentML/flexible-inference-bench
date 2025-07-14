import random
import os 
import numpy as np
import pytest
from transformers import AutoTokenizer
import json
import flexible_inference_benchmark.engine.data as data
import flexible_inference_benchmark.engine.distributions as distributions
from flexible_inference_benchmark.main import parse_args
from sharegpt_data import SHAREGPT_DATA

@pytest.mark.parametrize("ignore_input_distribution", [True, False])
def test_random_with_prefix_str(ignore_input_distribution):
    random.seed(0)
    np.random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    distribution = distributions.Same(7)
    output_token_distribution = distributions.Same(10)
    random_data_generator = data.Random.with_prefix_str(
        prefix_str="Hello world!", prefill_distribution=distribution, output_token_distribution=output_token_distribution, tokenizer=tokenizer, ignore_input_distribution=ignore_input_distribution
    )
    num_samples = 10
    output_data = random_data_generator.generate_data(num_samples)
    assert len(output_data) == num_samples
    for i in range(num_samples):
        assert len(output_data[i]) == 3
        sample_text, sample_len, sample_output_len = output_data[i]
        assert sample_output_len == 10
        if ignore_input_distribution:
            assert sample_text == "Hello world!"
            assert sample_len == 3
        else:
            assert sample_text.startswith("Hello world!")
            assert sample_text != "Hello world!"
            assert sample_len == 7

@pytest.mark.parametrize("ignore_input_distribution", [True, False])
def test_random_with_prefix_len(ignore_input_distribution):
    random.seed(0)
    np.random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    distribution = distributions.UniformInt(10, 11)
    output_token_distribution = distributions.UniformInt(10, 11)
    random_data_generator = data.Random.with_prefix_len(
        prefix_len=5, prefill_distribution=distribution, output_token_distribution=output_token_distribution, tokenizer=tokenizer, ignore_input_distribution=ignore_input_distribution
    )
    random_data = np.array(random_data_generator.generate_data(10))
    print(ignore_input_distribution, random_data)
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

def test_num_trials_cli_argument():
    """Test that num_trials CLI argument is properly parsed and validated."""
    import sys
    
    # Test default value
    original_argv = sys.argv
    try:
        sys.argv = ['fib', 'benchmark', '--model', 'test', '--base-url', 'http://test']
        args = parse_args()
        assert args.num_trials == 10
        
        # Test custom value
        sys.argv = ['fib', 'benchmark', '--model', 'test', '--base-url', 'http://test', '--num-trials', '5']
        args = parse_args()
        assert args.num_trials == 5
        
        # Test validation - zero value should fail
        sys.argv = ['fib', 'benchmark', '--model', 'test', '--base-url', 'http://test', '--num-trials', '0']
        with pytest.raises(SystemExit):
            parse_args()
            
        # Test validation - negative value should fail
        sys.argv = ['fib', 'benchmark', '--model', 'test', '--base-url', 'http://test', '--num-trials', '-1']
        with pytest.raises(SystemExit):
            parse_args()
            
    finally:
        sys.argv = original_argv
