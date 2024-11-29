import numpy as np
from abc import ABC, abstractmethod


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
    def mat(self):
        return self._mat

    # we do this to automatically add orbital dimensions if not already done
    # @jit(nopython=True, cache=True)
    @mat.setter
    def mat(self, value: np.ndarray | list):
        value = np.array(value, dtype=np.complex64) if isinstance(value, list) else value

        if (
            self._mat is None and len(value.shape) == self._num_frequency_dimensions
        ):  # i.e., if the object does not yet have orbital dimensions
            value = value.reshape((1,) * self._num_orbital_dimensions + value.shape)

        self._mat = value

    @property
    def current_shape(self):
        return self._mat.shape

    @property
    def original_shape(self):
        return self._original_shape

    @original_shape.setter
    def original_shape(self, value):
        self._original_shape = value

    @property
    def niv(self):
        return self.original_shape[-1] // 2 if self.full_niv_range else self.original_shape[-1]

    @property
    def full_niv_range(self):
        return self._full_niv_range

    @property
    def n_bands(self):
        return self.original_shape[0]

    @abstractmethod
    def to_compound_indices(self):
        pass

    @abstractmethod
    def to_full_indices(self):
        pass
