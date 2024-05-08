from dataclasses import dataclass
import abc
import logging
from typing import Any, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Distribution(abc.ABC):
    @abc.abstractmethod
    def generate_distribution(self, *args: Any) -> List[Any]:
        pass


@dataclass
class Poisson(Distribution):
    rate: float

    def generate_distribution(self, size: int) -> List[float]:
        logger.debug(f"Generating Poisson distribution of size {size} with rate {self.rate}")
        rval = np.zeros(size)
        scale = 1 / self.rate
        for i in range(1, size):
            rval[i] = rval[i - 1] + np.random.exponential(scale)
        return list(rval)


@dataclass
class Exponential(Distribution):
    rate: float

    def generate_distribution(self, size: int) -> List[float]:
        logger.debug(f"Generating Exponential distribution of size {size} with rate {self.rate}")
        return list(np.random.exponential(self.rate, size))


@dataclass
class UniformInt(Distribution):
    low: int
    high: int

    def generate_distribution(self, size: int) -> List[int]:
        logger.debug(f"Generating uniform int distribution of size {size} with low {self.low} and high {self.high}")
        return [int(elem) for elem in np.random.randint(self.low, self.high, size)]


@dataclass
class NormalInt(Distribution):
    mean: float
    std: float

    def generate_distribution(self, size: int) -> List[int]:
        logger.debug(f"Generating normal int distribution of size {size} with mean {self.mean} and std {self.std}")
        return [int(elem) for elem in np.random.normal(self.mean, self.std, size)]


@dataclass
class Same(Distribution):
    start: float

    def generate_distribution(self, size: int) -> List[float]:
        logger.debug(f"Generating same distribution of size {size} with start {self.start}")
        rval = np.ones(size) * self.start
        return list(rval)


@dataclass
class Even(Distribution):
    rate: float

    def generate_distribution(self, size: int) -> List[float]:
        logger.debug(f"Generating even distribution of size {size} with rate {self.rate}")
        return list(np.linspace(0, (size - 1) / self.rate, num=size))


@dataclass
class AdjustedUniformInt(Distribution):
    low: int
    high: int

    def generate_distribution(self, lengths: List[int]) -> List[int]:
        logging.debug(f"Generating adjusted uniform int distribution with low {self.low} and high {self.high}")
        rval = np.empty(len(lengths), dtype=np.int64)
        for i, length in enumerate(lengths):
            rval[i] = np.random.randint(self.low, self.high - length)
        return list(rval)


DISTRIBUTION_CLASSES = {
    "poisson": Poisson,
    "exponential": Exponential,
    "uniform": UniformInt,
    "normal": NormalInt,
    "same": Same,
    "even": Even,
}
