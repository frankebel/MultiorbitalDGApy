from abc import ABC
from copy import deepcopy

import numpy as np


class IHaveMat(ABC):
    def __init__(self, mat: np.ndarray):
        self._mat = mat.astype(np.complex64)

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray) -> None:
        self._mat = value

    def __mul__(self, other) -> "IHaveMat":
        if not isinstance(other, int | float | complex):
            raise ValueError("Multiplication only supported with numbers.")

        copy = deepcopy(self)
        copy.mat *= other
        return copy

    def __rmul__(self, other) -> "IHaveMat":
        return self.__mul__(other)

    def __neg__(self) -> "IHaveMat":
        return self.__mul__(-1.0)

    def __truediv__(self, other) -> "IHaveMat":
        return self.__mul__(1.0 / other)

    def __getitem__(self, item):
        return self.mat[item]

    def __setitem__(self, key, value):
        self.mat[key] = value
