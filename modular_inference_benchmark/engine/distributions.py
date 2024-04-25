from dataclasses import dataclass
import abc
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class Distribution(abc.ABC):
    @abc.abstractmethod
    def generate_distribution(self, *args: Any) -> npt.NDArray[Any]:
        pass


@dataclass
class Poisson(Distribution):
    rate: float

    def generate_distribution(self, size: int) -> npt.NDArray[np.float64]:
        logger.info(f"Generating Poisson distribution of size {size} with rate {self.rate}")
        rval = np.zeros(size)
        scale = 1 / self.rate
        for i in range(1, size):
            rval[i] = rval[i - 1] + np.random.exponential(scale)
        return rval

@dataclass
class Exponential(Distribution):
    rate: float 
    def generate_distribution(self, size: int) -> npt.NDArray[np.float64]:
        logger.info(f"Generating Exponential distribution of size {size} with rate {self.rate}")
        return np.random.exponential(self.rate, size)

@dataclass
class UniformInt(Distribution):
    low: int
    high: int

    def generate_distribution(self, size: int) -> npt.NDArray[np.int64]:
        logger.info(f"Generating uniform int distribution of size {size} with low {self.low} and high {self.high}")
        return np.random.randint(self.low, self.high, size)  # high is exclusive


@dataclass
class NormalInt(Distribution):
    mean: float
    std: float

    def generate_distribution(self, size: int) -> npt.NDArray[np.int64]:
        logger.info(f"Generating normal int distribution of size {size} with mean {self.mean} and std {self.std}")
        return np.random.normal(self.mean, self.std, size).astype(int)


@dataclass
class Same(Distribution):
    start: float

    def generate_distribution(self, size: int) -> npt.NDArray[np.float64]:
        logger.info(f"Generating same distribution of size {size} with start {self.start}")
        return np.ones(size) * self.start


@dataclass
class Even(Distribution):
    rate: float

    def generate_distribution(self, size: int) -> npt.NDArray[np.float64]:
        logger.info(f"Generating even distribution of size {size} with rate {self.rate}")
        return np.linspace(0, (size - 1) / self.rate, num=size)


@dataclass
class AdjustedUniformInt(Distribution):
    low: int
    high: int

    def generate_distribution(self, lengths: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        logger.info(f"Generating adjusted uniform int distribution with low {self.low} and high {self.high}")
        rval = np.empty(len(lengths))
        for i, length in enumerate(lengths):
            rval[i] = np.random.randint(self.low, self.high - length)
        return rval

DISTRIBUTION_CLASSES = {
    "poisson": Poisson,
    "exponential": Exponential,
    "uniform" : UniformInt,
    "normal": NormalInt,
    "same" : Same,
    "even" : Even,
    "adjusted-uniform": AdjustedUniformInt # ASK HOW DOES THIS WORKS AND UNDER WHICH SCENARIO THIS WOULD BE USEFUL
}