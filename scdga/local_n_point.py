import os

from scdga.n_point_base import *


class LocalNPoint(IHaveMat):
    """
    Base class for all (Local)NPoint objects, such as the (Full/Irreducible) Vertex functions, Susceptibilities,
    Fermi-Bose Vertices, Green's Function, Self-Energy and the like.
    """

    def __init__(
        self,
        mat: np.ndarray,
        num_orbital_dimensions: int,
        num_wn_dimensions: int,
        num_vn_dimensions: int,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
    ):
        IHaveMat.__init__(self, mat)

        assert num_orbital_dimensions in (2, 4), "2 or 4 orbital dimensions are supported."
        self._num_orbital_dimensions = num_orbital_dimensions

        assert num_vn_dimensions in (0, 1, 2), "0 - 2 fermionic frequency dimensions are supported."
        self._num_vn_dimensions = num_vn_dimensions

        assert num_wn_dimensions in (0, 1), "0 or 1 bosonic frequency dimensions are supported."
        self._num_wn_dimensions = num_wn_dimensions

        self._full_niv_range = full_niv_range
        self._full_niw_range = full_niw_range

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands.
        """
        return self.original_shape[0]

    @property
    def num_orbital_dimensions(self) -> int:
        """
        Returns the number of orbital dimensions.
        """
        return self._num_orbital_dimensions

    @property
    def num_wn_dimensions(self) -> int:
        """
        Returns the number of bosonic frequency dimensions.
        """
        return self._num_wn_dimensions

    @property
    def num_vn_dimensions(self) -> int:
        """
        Returns the number of fermionic frequency dimensions.
        """
        return self._num_vn_dimensions

    @property
    def niw(self) -> int:
        """
        Returns the number of bosonic frequencies in the object.
        """
        if self.num_wn_dimensions == 0:
            return 0
        axis = -(self.num_wn_dimensions + self.num_vn_dimensions)
        return self.original_shape[axis] // 2

    @property
    def niv(self) -> int:
        """
        Returns the number of fermionic frequencies in the object.
        """
        if self.num_vn_dimensions == 0:
            return 0
        return self.original_shape[-1] // 2

    @property
    def full_niw_range(self) -> bool:
        """
        Specifies whether the object is stored in the full bosonic frequency range or
        only a subset of it (only w >= 0).
        """
        return self._full_niw_range

    @property
    def full_niv_range(self) -> bool:
        """
        Specifies whether the object is stored in the full fermionic frequency range or
        only a subset of it (only v > 0).
        """
        return self._full_niv_range

    def cut_niw(self, niw_cut: int):
        """
        Allows to place a cutoff on the number of bosonic frequencies of the object.
        """
        if self.num_wn_dimensions == 0:
            raise ValueError("Cannot cut bosonic frequencies if there are none.")

        if niw_cut > self.niw:
            raise ValueError("Cannot cut more bosonic frequencies than the object has.")

        copy = deepcopy(self)

        niw_slice = slice(copy.niw - niw_cut, copy.niw + niw_cut + 1) if copy.full_niw_range else slice(0, niw_cut)

        if copy.num_vn_dimensions == 2:
            copy.mat = copy.mat[..., niw_slice, :, :]
        elif copy.num_vn_dimensions == 1:
            copy.mat = copy.mat[..., niw_slice, :]
        else:  # copy.num_vn_dimensions == 0
            copy.mat = copy.mat[..., niw_slice]

        copy.update_original_shape()
        return copy

    def cut_niv(self, niv_cut: int):
        """
        Allows to place a cutoff on the number of fermionic frequencies of the object.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("Cannot cut fermionic frequencies if there are none.")

        if niv_cut > self.niv:
            raise ValueError("Cannot cut more fermionic frequencies than the object has.")

        copy = deepcopy(self)

        niv_slice = slice(copy.niv - niv_cut, copy.niv + niv_cut) if copy.full_niv_range else slice(0, niv_cut)

        if copy.num_vn_dimensions == 2:
            copy.mat = copy.mat[..., niv_slice, niv_slice]
        elif copy.num_vn_dimensions == 1:
            copy.mat = copy.mat[..., niv_slice]

        copy.update_original_shape()
        return copy

    def cut_niw_and_niv(self, niw_cut: int, niv_cut: int):
        """
        Allows to place a cutoff on the number of bosonic and fermionic frequencies of the object.
        """
        return self.cut_niw(niw_cut).cut_niv(niv_cut)

    def extend_vn_to_diagonal(self):
        """
        Extends an object [...,w,v] to [...,w,v,v] by making a diagonal from the last dimension if the number of fermionic
        frequency dimensions is one.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("No fermionic frequency dimensions available for extension.")
        if self.num_vn_dimensions == 2:
            return self
        self.mat = np.einsum("...i,ij->...ij", self.mat, np.eye(self.current_shape[-1]), optimize=True)
        self._num_vn_dimensions = 2
        self.update_original_shape()
        return self

    def take_vn_diagonal(self):
        """
        Compresses an object [...w,v,v] to [...,w,v] by taking the diagonal of the last two dimensions.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("No fermionic frequency dimensions available for compression.")
        if self.num_vn_dimensions == 1:
            return self
        self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
        self._num_vn_dimensions = 1
        self.update_original_shape()
        return self

    def to_full_niw_range(self):
        """
        Converts the object to the full bosonic frequency range in-place.
        """
        if self.num_wn_dimensions == 0 or self.full_niw_range:
            return self

        niw_axis = -(self.num_wn_dimensions + self.num_vn_dimensions)
        ind = np.arange(1, self.current_shape[niw_axis])
        freq_axis = niw_axis
        trailing = "w"
        if self.num_vn_dimensions == 1:
            freq_axis = niw_axis, -1
            trailing = "wv"
        if self.num_vn_dimensions == 2:
            freq_axis = niw_axis, -2, -1
            trailing = "wvp"
        self.mat = np.concatenate(
            (
                np.einsum(
                    f"...abcd{trailing}->...dcba{trailing}",
                    np.conj(np.flip(np.take(self.mat, ind, axis=niw_axis), freq_axis)),
                ),
                self.mat,
            ),
            axis=niw_axis,
        )
        self.update_original_shape()
        self._full_niw_range = True
        return self

    def to_half_niw_range(self):
        """
        Converts the object to the half bosonic frequency range in-place.
        """
        if self.num_wn_dimensions == 0 or not self.full_niw_range:
            return self

        axis = -(self.num_wn_dimensions + self.num_vn_dimensions)
        ind = np.arange(self.current_shape[axis] // 2, self.current_shape[axis])
        self.mat = np.take(self.mat, ind, axis=axis)
        self.update_original_shape()
        self._full_niw_range = False
        return self

    def to_full_niv_range(self):
        """
        Converts the object to the full fermionic frequency range in-place. Works only on objects
        with a single fermionic frequency dimension.
        """
        if self.num_vn_dimensions == 0 or self.full_niv_range:
            return self

        if self.num_vn_dimensions != 1:
            raise ValueError("Can only convert to full niv range if the number of fermionic frequency dimensions is 1.")

        self.mat = np.concatenate((np.conj(np.flip(self.mat, axis=-1)), self.mat), axis=-1)
        self.update_original_shape()
        self._full_niv_range = True
        return self

    def to_half_niv_range(self):
        """
        Converts the object to the half fermionic frequency range in-place. Works only on objects
        with a single fermionic frequency dimension.
        """
        if self.num_vn_dimensions == 0 or not self.full_niv_range:
            return self

        if self.num_vn_dimensions != 1:
            raise ValueError("Can only convert to half niv range if the number of fermionic frequency dimensions is 1.")

        ind = np.arange(self.current_shape[-1] // 2, self.current_shape[-1])
        self.mat = np.take(self.mat, ind, axis=-1)
        self.update_original_shape()
        self._full_niv_range = False
        return self

    def flip_frequency_axis(self, axis: tuple | int):
        """
        Flips the matrix along the specified axis.
        """
        if self.num_wn_dimensions + self.num_vn_dimensions == 0:
            raise ValueError("Cannot flip the matrix if there are no frequency dimensions.")

        if isinstance(axis, int):
            axis = (axis,)

        axis_possible = tuple(range(-self.num_wn_dimensions - self.num_vn_dimensions, 0))
        if not set(axis).issubset(axis_possible):
            raise ValueError(f"Invalid axis {axis}. Possible axes are {axis_possible}.")

        self.mat = np.flip(self.mat, axis=axis)
        return self

    def save(self, output_dir: str = "./", name: str = "please_give_me_a_name") -> None:
        """
        Saves the content of the matrix to a file. Always saves it with half the niw range.
        """
        is_self_full_niw_range = self.full_niw_range
        np.save(os.path.join(output_dir, f"{name}.npy"), self.to_half_niw_range().mat, allow_pickle=False)
        if is_self_full_niw_range:
            self.to_full_niw_range()

    def _align_frequency_dimensions_for_operation(self, other: "LocalNPoint"):
        """
        Adapts the frequency dimensions of two LocalNPoint objects to fit each other for addition or multiplication.
        """
        self_extended = False
        other_extended = False
        if self.num_vn_dimensions == 1 and other.num_vn_dimensions == 2:
            self.extend_vn_to_diagonal()
            self_extended = True
        if self.num_vn_dimensions == 2 and other.num_vn_dimensions == 1:
            other = other.extend_vn_to_diagonal()
            other_extended = True
        return other, self_extended, other_extended
