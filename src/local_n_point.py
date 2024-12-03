import numpy as np

from i_have_mat import IHaveMat


class LocalNPoint(IHaveMat):
    def __init__(
        self,
        mat: np.ndarray,
        num_orbital_dimensions: int,
        num_bosonic_frequency_dimensions: int,
        num_fermionic_frequency_dimensions: int,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
    ):
        IHaveMat.__init__(self, mat)

        assert num_orbital_dimensions in (2, 4), "2 or 4 orbital dimensions are supported."
        self._num_orbital_dimensions = num_orbital_dimensions

        assert num_fermionic_frequency_dimensions in (0, 1, 2), "0 - 2 fermionic frequency dimensions are supported."
        self._num_fermionic_frequency_dimensions = num_fermionic_frequency_dimensions

        assert num_bosonic_frequency_dimensions in (0, 1), "0 or 1 bosonic frequency dimensions are supported."
        self._num_bosonic_frequency_dimensions = num_bosonic_frequency_dimensions

        self._full_niv_range = full_niv_range
        self._full_niw_range = full_niw_range

        self.original_shape = self.mat.shape

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
        if self._num_bosonic_frequency_dimensions == 0:
            return 0
        axis = -(self._num_bosonic_frequency_dimensions + self._num_fermionic_frequency_dimensions)
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

    def to_compound_indices(self) -> "LocalNPoint":
        if len(self.current_shape) == 3:  # [w,x1,x2]
            return self

        self.original_shape = self.mat.shape

        if self._num_bosonic_frequency_dimensions == 1:
            if self._num_fermionic_frequency_dimensions == 1:  # [o1,o2,o3,o4,w,v]
                self.mat = np.einsum("...i,ij->...ij", self.mat, np.eye(2 * self.niv))
            self.mat = self.mat.transpose(4, 0, 1, 5, 2, 3, 6).reshape(
                2 * self.niw + 1, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
            )
            return self

        raise ValueError(f"Converting to compound indices with shape {self.current_shape} not supported.")

    def to_full_indices(self, shape: tuple = None) -> "LocalNPoint":
        if (
            len(self.current_shape)
            == self._num_orbital_dimensions
            + self._num_bosonic_frequency_dimensions
            + self._num_fermionic_frequency_dimensions
        ):
            return self
        elif len(self.current_shape) == 3:  # [w,x1,x2]
            self.original_shape = shape if shape is not None else self.original_shape
            compound_index_shape = (self.n_bands, self.n_bands, 2 * self.niv)

            if self._num_bosonic_frequency_dimensions == 1:
                self.mat = (self.mat.reshape((2 * self.niw + 1,) + compound_index_shape * 2)).transpose(
                    1, 2, 4, 5, 0, 3, 6
                )
                if self._num_fermionic_frequency_dimensions == 1:  # original was [o1,o2,o3,o4,w,v]
                    self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
                return self
        else:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")

    def invert(self) -> "LocalNPoint":
        copy = self
        copy = copy.to_compound_indices()
        copy.mat = np.linalg.inv(copy.mat)
        return copy.to_full_indices()

    def __invert__(self) -> "LocalNPoint":
        return self.invert()
