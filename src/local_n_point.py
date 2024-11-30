from abc import ABC, abstractmethod

import numpy as np


class LocalNPoint(ABC):
    def __init__(
        self,
        mat: np.ndarray,
        num_orbital_dimensions: int,
        num_frequency_dimensions: int,
        full_niv_range: bool,
    ):
        self._num_orbital_dimensions = num_orbital_dimensions
        self._num_frequency_dimensions = num_frequency_dimensions
        self._full_niv_range = full_niv_range

        self._mat = None
        self.mat: np.ndarray = mat
        self.original_shape = self.mat.shape

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray) -> None:
        self._mat = value.astype(np.complex64)

    @property
    def current_shape(self) -> tuple:
        return self._mat.shape

    @property
    def original_shape(self) -> tuple:
        return self._original_shape

    @original_shape.setter
    def original_shape(self, value) -> None:
        self._original_shape = value

    @property
    def niv(self) -> int:
        return self.original_shape[-1] // 2 if self.full_niv_range else self.original_shape[-1]

    @property
    def full_niv_range(self) -> bool:
        return self._full_niv_range

    @property
    def n_bands(self) -> int:
        return self.original_shape[0]

    @abstractmethod
    def to_compound_indices(self) -> "LocalNPoint":
        pass

    @abstractmethod
    def to_full_indices(self, shape: tuple = None) -> "LocalNPoint":
        pass

    def __getitem__(self, item):
        return self.mat[item]

    def __setitem__(self, key, value):
        self.mat[key] = value
