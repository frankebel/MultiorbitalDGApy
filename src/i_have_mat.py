import numpy as np
from abc import ABC


class IHaveMat(ABC):
    def __init__(self, mat: np.ndarray):
        self._mat = mat

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray) -> None:
        self._mat = value.astype(np.complex64)

    def __mul__(self, other: int | float | complex) -> "IHaveMat":
        if not isinstance(other, int | float | complex):
            raise ValueError("Other needs to be a number.")

        copy = self
        copy.mat *= other
        return copy

    def __rmul__(self, other: int | float | complex) -> "IHaveMat":
        return self.__mul__(other)

    def __getitem__(self, item):
        return self.mat[item]

    def __setitem__(self, key, value):
        self.mat[key] = value
