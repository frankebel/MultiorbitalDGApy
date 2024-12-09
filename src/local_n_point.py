import os
from copy import deepcopy

import numpy as np

from i_have_mat import IHaveMat


class LocalNPoint(IHaveMat):
    """
    Base class for all LocalNPoint objects, such as the (Full/Irreducible) Vertex functions, Susceptibilities,
    Fermi-Bose Vertices, Green's Function, Self-Energy and the like.
    """

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
    def num_orbital_dimensions(self) -> int:
        return self._num_orbital_dimensions

    @property
    def num_bosonic_frequency_dimensions(self) -> int:
        return self._num_bosonic_frequency_dimensions

    @property
    def num_fermionic_frequency_dimensions(self) -> int:
        return self._num_fermionic_frequency_dimensions

    @property
    def niw(self) -> int:
        """
        Returns the number of bosonic frequencies in the object.
        """
        if self.num_bosonic_frequency_dimensions == 0:
            return 0
        axis = -(self.num_bosonic_frequency_dimensions + self.num_fermionic_frequency_dimensions)
        return self.original_shape[axis] // 2 if self.full_niv_range else self.original_shape[axis]

    @property
    def niv(self) -> int:
        """
        Returns the number of fermionic frequencies in the object.
        """
        if self.num_fermionic_frequency_dimensions == 0:
            return 0
        return self.original_shape[-1] // 2 if self.full_niv_range else self.original_shape[-1]

    @property
    def full_niw_range(self) -> bool:
        """
        Specifies whether the object is stored in the full bosonic frequency range or
        only a subset of it (e.g. only w >= 0).
        """
        return self._full_niw_range

    @property
    def full_niv_range(self) -> bool:
        """
        Specifies whether the object is stored in the full fermionic frequency range or
        only a subset of it (e.g. only w >= 0).
        """
        return self._full_niv_range

    def cut_niw(self, niw_cut: int) -> "LocalNPoint":
        """
        Allows to place a cutoff on the number of bosonic frequencies in the object.
        Cuts all bosonic frequency dimensions, modifies and returns the original object without creating a copy.
        """
        if self.num_bosonic_frequency_dimensions == 0:
            raise ValueError("Cannot cut bosonic frequencies if there are none.")

        if niw_cut > self.niw:
            raise ValueError("Cannot cut more bosonic frequencies than the object has.")

        if self.full_niw_range:
            if self.num_fermionic_frequency_dimensions == 2:
                self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1, :, :]
            elif self.num_fermionic_frequency_dimensions == 1:
                self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1, :]
            elif self.num_fermionic_frequency_dimensions == 0:
                self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1]
        else:
            if self.num_fermionic_frequency_dimensions == 2:
                self.mat = self.mat[..., :niw_cut, :, :]
            elif self.num_fermionic_frequency_dimensions == 1:
                self.mat = self.mat[..., :niw_cut, :]
            elif self.num_fermionic_frequency_dimensions == 0:
                self.mat = self.mat[..., :niw_cut]

        self.original_shape = self.mat.shape
        return self

    def cut_niv(self, niv_cut: int) -> "LocalNPoint":
        """
        Allows to place a cutoff on the number of fermionic frequencies in the object.
        Cuts all fermionic frequency dimensions, modifies and returns the original object without creating a copy.
        """
        if self.num_fermionic_frequency_dimensions == 0:
            raise ValueError("Cannot cut fermionic frequencies if there are none.")

        if niv_cut > self.niv:
            raise ValueError("Cannot cut more fermionic frequencies than the object has.")

        if self.full_niv_range:
            if self.num_fermionic_frequency_dimensions == 2:
                self.mat = self.mat[
                    ..., self.niv - niv_cut : self.niv + niv_cut, self.niv - niv_cut : self.niv + niv_cut
                ]
            elif self.num_fermionic_frequency_dimensions == 1:
                self.mat = self.mat[..., self.niv - niv_cut : self.niv + niv_cut]
        else:
            if self.num_fermionic_frequency_dimensions == 2:
                self.mat = self.mat[..., :niv_cut, :niv_cut]
            elif self.num_fermionic_frequency_dimensions == 1:
                self.mat = self.mat[..., :niv_cut]

        self.original_shape = self.mat.shape
        return self

    def cut_niw_and_niv(self, niw_cut: int, niv_cut: int) -> "LocalNPoint":
        """
        Allows to place a cutoff on the number of bosonic and fermionic frequencies in the object.
        Cuts all fermionic and bosonic frequency dimensions, modifies
        and returns the original object without creating a copy.
        """
        return self.cut_niw(niw_cut).cut_niv(niv_cut)

    def to_compound_indices(self) -> "LocalNPoint":
        """
        Converts a fermionic frequency axis and two orbital axis into a single compound index. This is useful for
        multiplications between and inversions of LocalNPoint objects. Depending on the number of fermionic frequency
        dimensions, other strategies are used to form the compound indices.
        """
        if len(self.current_shape) == 3:  # [w,x1,x2]
            return self

        if self.num_bosonic_frequency_dimensions != 1:
            raise ValueError(f"Converting to compound indices with shape {self.current_shape} not supported.")

        self.original_shape = self.mat.shape

        if self.num_fermionic_frequency_dimensions == 0:  # [o1,o2,o3,o4,w]
            self.mat = self.mat.transpose(4, 0, 1, 2, 3).reshape(2 * self.niw + 1, self.n_bands**2, self.n_bands**2)
            return self

        if self.num_fermionic_frequency_dimensions == 1:  # [o1,o2,o3,o4,w,v]
            self.extend_last_frequency_axis_to_diagonal()

        self.mat = self.mat.transpose(4, 0, 1, 5, 2, 3, 6).reshape(
            2 * self.niw + 1, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
        )
        return self

    def to_full_indices(self, shape: tuple = None) -> "LocalNPoint":
        """
        Converts an object stored with compound indices to an object that has unraveled orbital and frequency axis.
        This needs to be done to have the object in the original shape present before the transformation to compound indices.
        """
        if (
            len(self.current_shape)
            == self.num_orbital_dimensions
            + self.num_bosonic_frequency_dimensions
            + self.num_fermionic_frequency_dimensions
        ):
            return self
        elif len(self.current_shape) == 3:  # [w,x1,x2]
            self.original_shape = shape if shape is not None else self.original_shape
            if self.num_bosonic_frequency_dimensions == 1:
                if self.num_fermionic_frequency_dimensions == 0:  # original was [o1,o2,o3,o4,w]
                    self.mat = self.mat.reshape(
                        (2 * self.niw + 1,) + (self.n_bands,) * self.num_orbital_dimensions
                    ).transpose(1, 2, 3, 4, 0)
                    return self

                compound_index_shape = (self.n_bands, self.n_bands, 2 * self.niv)

                self.mat = (self.mat.reshape((2 * self.niw + 1,) + compound_index_shape * 2)).transpose(
                    1, 2, 4, 5, 0, 3, 6
                )

                if self.num_fermionic_frequency_dimensions == 1:  # original was [o1,o2,o3,o4,w,v]
                    self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
                return self
        else:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")

    def invert(self) -> "LocalNPoint":
        """
        Inverts the LocalNPoint object by transforming it to compound indices.
        """
        copy = deepcopy(self)
        copy = copy.to_compound_indices()
        copy.mat = np.linalg.inv(copy.mat)
        return copy.to_full_indices()

    def __invert__(self) -> "LocalNPoint":
        return self.invert()

    def extend_last_frequency_axis_to_diagonal(self) -> "LocalNPoint":
        """
        Extends an object [...,w,v] to [...,w,v,v] by making a diagonal from the last dimension.
        """
        if self.num_fermionic_frequency_dimensions == 2:
            raise ValueError("Extending to three or more fermionic frequency dimensions is not supported.")
        self.mat = np.einsum("...i,ij->...ij", self.mat, np.eye(self.mat.shape[-1]))
        self._num_fermionic_frequency_dimensions += 1
        self.original_shape = self.current_shape
        return self

    def compress_last_two_frequency_dimensions_to_single_dimension(self) -> "LocalNPoint":
        """
        Compresses an object [...w,v,v] to [...,w,v] by taking the diagonal of the last two dimensions.
        """
        if self.num_fermionic_frequency_dimensions < 2:
            raise ValueError("Cannot compress two fermionic frequency dimensions to one if there are less than two.")
        self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
        self._num_fermionic_frequency_dimensions -= 1
        self.original_shape = self.current_shape
        return self

    def save(self, output_dir: str = "./", name: str = "please_give_me_a_name"):
        """
        Saves the content of the matrix to a file.
        """
        np.save(os.path.join(output_dir, f"{name}.npy"), self.mat, allow_pickle=True)
