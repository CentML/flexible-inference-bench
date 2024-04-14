import numpy as np

import modular_inference_benchmark.engine.distributions as distributions


def test_poisson_len():
    poisson = distributions.Poisson(1.0)
    generated = poisson.generate_distribution(10)
    assert len(generated) == 10


def test_uniform_int_len():
    uniform_int = distributions.UniformInt(0, 10)
    generated = uniform_int.generate_distribution(10)
    assert len(generated) == 10


def test_normal_int_len():
    normal_int = distributions.NormalInt(0.0, 1.0)
    generated = normal_int.generate_distribution(10)
    assert len(generated) == 10


def test_same_len():
    same = distributions.Same(0.0)
    generated = same.generate_distribution(10)
    assert len(generated) == 10


def test_even_len():
    even = distributions.Even(1.0)
    generated = even.generate_distribution(10)
    assert len(generated) == 10


def test_adjusted_uniform_int_len():
    adjusted_uniform_int = distributions.AdjustedUniformInt(0, 10)
    generated = adjusted_uniform_int.generate_distribution(np.zeros(10))
    assert len(generated) == 10


def test_poisson_values():
    poisson = distributions.Poisson(1.0)
    generated = poisson.generate_distribution(10)
    assert np.all(generated >= 0)


def test_uniform_int_values():
    uniform_int = distributions.UniformInt(0, 10)
    generated = uniform_int.generate_distribution(10)
    assert np.all(generated >= 0)
    assert np.all(generated < 10)


def test_same_values():
    same = distributions.Same(0.0)
    generated = same.generate_distribution(10)
    assert np.all(generated == 0.0)


def test_even_values():
    even = distributions.Even(1.0)
    generated = even.generate_distribution(10)
    assert np.all(generated >= 0) and np.all(generated <= 9)


def test_adjusted_uniform_int_values():
    adjusted_uniform_int = distributions.AdjustedUniformInt(0, 10)
    generated = adjusted_uniform_int.generate_distribution(np.zeros(10))
    assert np.all(generated >= 0)
    assert np.all(generated < 10)
