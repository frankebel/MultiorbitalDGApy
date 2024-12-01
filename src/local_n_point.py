from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Channel(Enum):
    DENS: str = "dens"
    MAGN: str = "magn"
    NONE: str = "none"


class LocalNPoint(ABC):
    def __init__(
        self,
        mat: np.ndarray,
        num_orbital_dimensions: int,
        num_bosonic_frequency_dimensions: int,
        num_fermionic_frequency_dimensions: int,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
    ):
        self._num_orbital_dimensions = num_orbital_dimensions
        self._num_fermionic_frequency_dimensions = num_fermionic_frequency_dimensions
        self._num_bosonic_frequency_dimensions = num_bosonic_frequency_dimensions
        self._num_frequency_dimensions = (
            self._num_bosonic_frequency_dimensions + self._num_fermionic_frequency_dimensions
        )
        self._full_niv_range = full_niv_range
        self._full_niw_range = full_niw_range

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
    def n_bands(self) -> int:
        return self.original_shape[0]

    @property
    def niw(self) -> int:
        if self._num_fermionic_frequency_dimensions == 0:
            return 0
        axis = -self._num_frequency_dimensions
        return self.original_shape[axis] // 2 if self.full_niv_range else self.original_shape[axis]

    @property
    def niv(self) -> int:
        if self._num_fermionic_frequency_dimensions == 0:
            return 0
        return self.original_shape[-1] // 2 if self.full_niv_range else self.original_shape[-1]

    @property
    def full_niw_range(self) -> bool:
        return self._full_niw_range

    @property
    def full_niv_range(self) -> bool:
        return self._full_niv_range

    def cut_niw(self, niw_cut: int) -> "LocalNPoint":
        if self._num_bosonic_frequency_dimensions == 0:
            raise ValueError("Cannot cut bosonic frequencies if there are none.")

        if niw_cut > self.niw:
            raise ValueError("Cannot cut more bosonic frequencies than the object has.")

        if self.full_niw_range:
            if self._num_fermionic_frequency_dimensions == 2:
                self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1, :, :]
            elif self._num_fermionic_frequency_dimensions == 1:
                self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1, :]
            elif self._num_fermionic_frequency_dimensions == 0:
                self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1]
        else:
            if self._num_fermionic_frequency_dimensions == 2:
                self.mat = self.mat[..., :niw_cut, :, :]
            elif self._num_fermionic_frequency_dimensions == 1:
                self.mat = self.mat[..., :niw_cut, :]
            elif self._num_fermionic_frequency_dimensions == 0:
                self.mat = self.mat[..., :niw_cut]

        self.original_shape = self.mat.shape
        return self

    def cut_niv(self, niv_cut: int) -> "LocalNPoint":
        if self._num_fermionic_frequency_dimensions == 0:
            raise ValueError("Cannot cut fermionic frequencies if there are none.")

        if niv_cut > self.niv:
            raise ValueError("Cannot cut more fermionic frequencies than the object has.")

        if self.full_niv_range:
            if self._num_fermionic_frequency_dimensions == 2:
                self.mat = self.mat[
                    ..., self.niv - niv_cut : self.niv + niv_cut, self.niv - niv_cut : self.niv + niv_cut
                ]
            elif self._num_fermionic_frequency_dimensions == 1:
                self.mat = self.mat[..., self.niv - niv_cut : self.niv + niv_cut]
        else:
            if self._num_fermionic_frequency_dimensions == 2:
                self.mat = self.mat[..., :niv_cut, :niv_cut]
            elif self._num_fermionic_frequency_dimensions == 1:
                self.mat = self.mat[..., :niv_cut]

        self.original_shape = self.mat.shape
        return self

    def cut_niw_and_niv(self, niw_cut: int, niv_cut: int) -> "LocalNPoint":
        return self.cut_niw(niw_cut).cut_niv(niv_cut)

    @abstractmethod
    def to_compound_indices(self) -> "LocalNPoint":
        pass

    @abstractmethod
    def to_full_indices(self, shape: tuple = None) -> "LocalNPoint":
        pass

    @abstractmethod
    def invert(self) -> "LocalNPoint":
        pass

    def __invert__(self) -> "LocalNPoint":
        return self.invert()

    def __getitem__(self, item):
        return self.mat[item]

    def __setitem__(self, key, value):
        self.mat[key] = value
