import numpy as np

from dataclasses import dataclass
import abc


@dataclass
class distribution(abc.ABC):
    pass

    @abc.abstractmethod
    def generate_distribution(self: int) -> np.ndarray:
        pass


@dataclass
class poisson:
    rate: float

    def generate_distribution(self, size: int) -> np.ndarray:
        rval = np.zeros(size)
        scale = 1 / self.rate
        for i in range(1, size):
            rval[i] = rval[i - 1] + np.random.exponential(scale)
        return rval


@dataclass
class uniformInt:
    low: int
    high: int

    def generate_distribution(self, size: int) -> np.ndarray:
        return np.random.randint(self.low, self.high, size)  # high is exclusive


@dataclass
class normalInt:
    mean: float
    std: float

    def generate_distribution(self, size: int) -> np.ndarray:
        return np.random.normal(self.mean, self.std, size).astype(int)


@dataclass
class same:
    start: float

    def generate_distribution(self, size: int) -> np.ndarray:
        return np.ones(size) * self.start


@dataclass
class even:
    rate: float

    def generate_distribution(self, size: int) -> np.ndarray:
        return np.linspace(0, (size - 1) / self.rate, num=size)
