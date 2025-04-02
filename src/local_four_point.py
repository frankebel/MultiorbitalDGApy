from matplotlib import pyplot as plt

from interaction import LocalInteraction
from local_n_point import LocalNPoint
from matsubara_frequencies import MFHelper
from n_point_base import *


class LocalFourPoint(LocalNPoint, IHaveChannel):
    """
    This class is used to represent a local four-point object in a given channel with a given number of bosonic and
    fermionic frequency dimensions. These were added to make matrix operations with other objects easier.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ):
        LocalNPoint.__init__(
            self,
            mat,
            4,
            num_wn_dimensions,
            num_vn_dimensions,
            full_niw_range,
            full_niv_range,
        )
        IHaveChannel.__init__(self, channel, frequency_notation)

    def __add__(self, other) -> "LocalFourPoint":
        """
        Addition for LocalFourPoint objects. Allows for A + B = C.
        """
        return self.add(other)

    def __radd__(self, other) -> "LocalFourPoint":
        """
        Addition for LocalFourPoint objects. Allows for A + B = C.
        """
        return self.add(other)

    def __sub__(self, other) -> "LocalFourPoint":
        """
        Subtraction for LocalFourPoint objects. Allows for A - B = C.
        """
        return self.sub(other)

    def __rsub__(self, other) -> "LocalFourPoint":
        """
        Subtraction for LocalFourPoint objects. Allows for A - B = C.
        """
        return self.sub(other)

    def __mul__(self, other) -> "LocalFourPoint":
        """
        Multiplication for LocalFourPoint objects. Allows for the multiplication with numbers, numpy arrays and
        other LocalFourPoint objects, such that A^{v} * B^{v'} = C^{vv'}.
        """
        return self.mul(other)

    def __rmul__(self, other) -> "LocalFourPoint":
        """
        Multiplication for LocalFourPoint objects. Allows for the multiplication with numbers, numpy arrays and
        other LocalFourPoint objects, such that A^{v} * B^{v'} = C^{vv'}.
        """
        return self.mul(other)

    def __matmul__(self, other) -> "LocalFourPoint":
        """
        Matrix multiplication for LocalFourPoint objects. Allows for A @ B = C using compound indices.
        """
        return self.matmul(other, left_hand_side=True)

    def __rmatmul__(self, other) -> "LocalFourPoint":
        """
        Matrix multiplication for LocalFourPoint objects. Allows for A @ B = C using compound indices.
        """
        return self.matmul(other, left_hand_side=False)

    def __invert__(self) -> "LocalFourPoint":
        """
        Inverts the LocalFourPoint object by transforming it to compound indices.
        """
        return self.invert()

    def __pow__(self, power, modulo=None):
        """
        Exponentiation for LocalFourPoint objects. Allows for A ** n = B, where n is an integer. If n < 0, then we
        exponentiate the inverse of A |n| times, i.e., A ** (-n) = A^(-1) ** n.
        """
        return self.power(power, LocalFourPoint.identity_like(self))

    def power(self, power: int, identity):
        """
        Exponentiation for LocalFourPoint objects. Allows for A ** n = B, where n is an integer. If n < 0, then we
        exponentiate the inverse of A |n| times, i.e., A ** (-n) = A^(-1) ** n. Requires the input of the identity.
        """
        if not isinstance(power, int):
            raise ValueError("Only integer powers are supported.")

        if power == 0:
            return identity
        if power == 1:
            return self
        if power < 0:
            return (~self) ** abs(power)

        result = identity
        base = deepcopy(self)

        # Exponentiation by squaring
        while power > 0:
            if power % 2 == 1:
                result @= base
            base @= base
            power //= 2

        return result

    def symmetrize_v_vp(self):
        """
        Symmetrize with respect to (v,v'). This is justified for SU(2) symmetric systems. (Thesis Rohringer p. 72)
        """
        self.mat = 0.5 * (self.mat + np.swapaxes(self.mat, -1, -2))
        return self

    def sum_over_orbitals(self, orbital_contraction: str = "abcd->ad"):
        """
        Sums over the given orbitals.
        """
        split = orbital_contraction.split("->")
        if len(split[0]) != 4 or len(split[1]) > len(split[0]):
            raise ValueError("Invalid orbital contraction.")

        self.mat = np.einsum(f"{split[0]}...->{split[1]}...", self.mat)
        diff = len(split[0]) - len(split[1])
        self.update_original_shape()
        self._num_orbital_dimensions -= diff
        return self

    def sum_over_vn(self, beta: float, axis: tuple = (-1,)):
        """
        Sums over specific fermionic frequency dimensions and multiplies with the correct prefactor 1/beta^(n_dim).
        """
        if len(axis) > self.num_vn_dimensions:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        copy_mat = 1 / beta ** len(axis) * np.sum(self.mat, axis=axis)
        self.update_original_shape()
        return LocalFourPoint(
            copy_mat,
            self.channel,
            self.num_wn_dimensions,
            self.num_vn_dimensions - len(axis),
            self.full_niw_range,
            self.full_niv_range,
        )

    def sum_over_all_vn(self, beta: float):
        """
        Sums over all fermionic frequency dimensions and multiplies with the correct prefactor 1/beta^(n_dim).
        """
        if self.num_vn_dimensions == 0:
            return self
        elif self.num_vn_dimensions == 1:
            axis = (-1,)
        elif self.num_vn_dimensions == 2:
            axis = (-2, -1)
        else:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        return self.sum_over_vn(beta, axis=axis)

    def contract_legs(self, beta: float):
        """
        Sums over all fermionic frequency dimensions if the object has 2 fermionic frequency dimensions and sums over
        the inner two orbitals.
        """
        if self.num_vn_dimensions != 2:
            raise ValueError("This method is only implemented for objects with 2 fermionic frequency dimensions.")
        return self.sum_over_vn(beta, axis=(-1, -2)).sum_over_orbitals("abcd->ad")

    def to_compound_indices(self) -> "LocalFourPoint":
        r"""
        Converts the indices of the LocalNPoint object

        .. math:: F^{wv(v')}_{lmm'l'}

        to compound indices

        .. math:: F^{w}_{c_1, c_2}.

        for a couple of shape cases. We group {v, l, m} and {v',m',l'} into two indices.

        .. math:: c_1 \;and\; c_2.
        """
        if len(self.current_shape) == 3:  # [w,x1,x2]
            return self

        if self.num_wn_dimensions != 1:
            raise ValueError(f"Cannot convert to compound indices if there are no bosonic frequencies.")

        self.update_original_shape()

        if self.num_vn_dimensions == 0:  # [o1,o2,o3,o4,w]
            self.mat = self.mat.transpose(4, 0, 1, 2, 3).reshape(2 * self.niw + 1, self.n_bands**2, self.n_bands**2)
            return self

        if self.num_vn_dimensions == 1:  # [o1,o2,o3,o4,w,v]
            self.extend_vn_to_diagonal()

        self.mat = self.mat.transpose(4, 0, 1, 5, 2, 3, 6).reshape(
            2 * self.niw + 1, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
        )
        return self

    def to_full_indices(self, shape: tuple = None) -> "LocalFourPoint":
        """
        Converts an object stored with compound indices to an object that has unraveled orbital and frequency axes.
        """
        if len(self.current_shape) == self.num_orbital_dimensions + self.num_wn_dimensions + self.num_vn_dimensions:
            return self

        if len(self.current_shape) != 3:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")

        if self.num_wn_dimensions != 1:
            raise ValueError("Number of bosonic frequency dimensions must be 1.")

        self.original_shape = shape if shape is not None else self.original_shape

        if self.num_vn_dimensions == 0:  # original was [o1,o2,o3,o4,w]
            self.mat = self.mat.reshape((2 * self.niw + 1,) + (self.n_bands,) * self.num_orbital_dimensions).transpose(
                1, 2, 3, 4, 0
            )
            return self

        compound_index_shape = (self.n_bands, self.n_bands, 2 * self.niv)

        self.mat = self.mat.reshape((2 * self.niw + 1,) + compound_index_shape * 2).transpose(1, 2, 4, 5, 0, 3, 6)

        if self.num_vn_dimensions == 1:  # original was [o1,o2,o3,o4,w,v]
            self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
        return self

    def invert(self):
        """
        Inverts the object by transforming it to compound indices. Returns the object always in half of their niw range.
        """
        copy = deepcopy(self).to_half_niw_range().to_compound_indices()
        copy.mat = np.linalg.inv(copy.mat)
        return copy.to_full_indices()

    def matmul(self, other, left_hand_side: bool = True) -> "LocalFourPoint":
        """
        Helper method that allows for matrix multiplication for LocalFourPoint objects. Depending on the
        number of frequency dimensions, the objects have to be multiplied differently.
        """
        if not isinstance(other, (LocalFourPoint, LocalInteraction)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, LocalInteraction):
            einsum_str = {
                0: "abijw,jief->abefw" if left_hand_side else "abij,jiefw->abefw",
                1: "abijwv,jief->abefwv" if left_hand_side else "abij,jiefwv->abefwv",
                2: "abijwvp,jief->abefwvp" if left_hand_side else "abij,jiefwvp->abefwvp",
            }.get(self.num_vn_dimensions)
            return LocalFourPoint(
                np.einsum(einsum_str, self.mat, other.mat, optimize=True),
                self.channel,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

        channel = self.channel if self.channel != SpinChannel.NONE else other.channel

        if (
            self.num_wn_dimensions == 1
            and other.num_wn_dimensions == 1
            and (self.num_vn_dimensions == 0 or other.num_vn_dimensions == 0)
        ):
            # special case if one of two LocalFourPoint object has no fermionic frequency dimensions
            # straightforward contraction is saving memory as we do not have to add fermionic frequency dimensions
            einsum_str = {
                (0, 2): "abcdw,dcefwvp->abefwvp",
                (0, 1): "abcdw,dcefwv->abefwv",
                (0, 0): "abcdw,dcefw->abefw",
                (1, 0): "abcdwv,dcefw->abefwv",
                (2, 0): "abcdwvp,dcefw->abefwvp",
            }.get((self.num_vn_dimensions, other.num_vn_dimensions))

            return LocalFourPoint(
                np.einsum(einsum_str, self.mat, other.mat, optimize=True),
                channel,
                self.num_wn_dimensions,
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                full_niw_range=self.full_niw_range,
                full_niv_range=self.full_niv_range,
            )

        self_full_niw_range = self.full_niw_range
        other_full_niw_range = other.full_niw_range

        # we do not use np.einsum here because it is faster to use np.matmul with compound indices instead of np.einsum
        self.to_half_niw_range().to_compound_indices()
        other = other.to_half_niw_range().to_compound_indices()
        # for __matmul__ self needs to be the LHS object, for __rmatmul__ self needs to be the RHS object
        new_mat = np.matmul(self.mat, other.mat) if left_hand_side else np.matmul(other.mat, self.mat)

        shape = (
            self.original_shape
            if self.num_vn_dimensions == max(self.num_vn_dimensions, other.num_vn_dimensions)
            else other.original_shape
        )

        self.to_full_indices()
        if self_full_niw_range:
            self.to_full_niw_range()
        other = other.to_full_indices()
        if other_full_niw_range:
            other = other.to_full_niw_range()

        return LocalFourPoint(
            new_mat,
            channel,
            self.num_wn_dimensions,
            max(self.num_vn_dimensions, other.num_vn_dimensions),
            full_niw_range=False,
            full_niv_range=self.full_niv_range,
        ).to_full_indices(shape)

    def mul(self, other):
        r"""
        Allows for the multiplication with a number, a numpy array or a LocalFourPoint object. In the latter instance,
        we require both objects to only have one niv dimension, such that

        .. math:: A_{abcd}^v * B_{abcd}^{v'} = C_{abcd}^{vv'}.
        """
        if not isinstance(other, (int, float, complex, np.ndarray, LocalFourPoint)):
            raise ValueError("Multiplication only supported with numbers, numpy arrays or LocalFourPoint objects.")

        if not isinstance(other, LocalFourPoint):
            copy = deepcopy(self)
            copy.mat *= other
            return copy

        if self.num_vn_dimensions != 1 or other.num_vn_dimensions != 1:
            raise ValueError("Both objects must have only one fermionic frequency dimension.")

        self.to_half_niw_range().to_half_niv_range()
        other = other.to_half_niw_range().to_half_niv_range()
        result_mat = self.times("abcdwv,abcdwp->abcdwvp", other)
        self.to_full_niw_range().to_full_niv_range()
        other = other.to_full_niw_range().to_full_niv_range()

        return (
            LocalFourPoint(result_mat, self.channel, 1, 2, False, False, self.frequency_notation)
            .to_full_niw_range()
            .to_full_niv_range()
        )

    def add(self, other):
        """
        Helper method that allows for addition of local FourPoint objects.
        Depending on the number of frequency and momentum dimensions, the objects have to be added differently.
        """
        if not isinstance(other, (LocalFourPoint, LocalInteraction, np.ndarray, float, int, complex)):
            raise ValueError(f"Operations '+/-' for {type(self)} and {type(other)} not supported.")

        if isinstance(other, (np.ndarray, float, int, complex)):
            return LocalFourPoint(
                self.mat + other,
                self.channel,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

        if isinstance(other, LocalInteraction):
            other_mat = other.mat.reshape(other.mat.shape + (1,) * (self.num_wn_dimensions + self.num_vn_dimensions))
            return LocalFourPoint(
                self.mat + other_mat,
                self.channel,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

        if self.num_wn_dimensions != other.num_wn_dimensions:
            raise ValueError("Number of bosonic frequency dimensions do not match.")

        self_full_niw_range = self.full_niw_range
        other_full_niw_range = other.full_niw_range

        self.to_half_niw_range()
        other = other.to_half_niw_range()

        channel = self.channel if self.channel != SpinChannel.NONE else other.channel

        if self.num_vn_dimensions == 0 or other.num_vn_dimensions == 0:
            self_mat = self.mat
            other_mat = other.mat

            if self.num_vn_dimensions != other.num_vn_dimensions:
                if self.num_vn_dimensions == 0:
                    self_mat = np.expand_dims(self_mat, axis=-1 if other.num_vn_dimensions == 1 else (-1, -2))
                elif other.num_vn_dimensions == 0:
                    other_mat = np.expand_dims(other_mat, axis=-1 if self.num_vn_dimensions == 1 else (-1, -2))

            result = LocalFourPoint(
                self_mat + other_mat,
                self.channel,
                max(self.num_wn_dimensions, other.num_wn_dimensions),
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

            if self_full_niw_range:
                self.to_full_niw_range()
            if other_full_niw_range:
                other = other.to_full_niw_range()

            return result

        other, self_extended, other_extended = self._align_frequency_dimensions_for_operation(other)

        result = LocalFourPoint(
            self.mat + other.mat,
            channel,
            self.num_wn_dimensions,
            max(self.num_vn_dimensions, other.num_vn_dimensions),
            self.full_niw_range,
            self.full_niv_range,
            self.frequency_notation,
        )

        if self_full_niw_range:
            self.to_full_niw_range()
        if other_full_niw_range:
            other = other.to_full_niw_range()

        other = self._revert_frequency_dimensions_after_operation(other, other_extended, self_extended)
        return result

    def sub(self, other):
        return self.add(-other)

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "LocalFourPoint":
        """
        Permutes the orbitals of the four-point object.
        """
        split = permutation.split("->")
        if (
            len(split) != 2
            or len(split[0]) != self.num_orbital_dimensions
            or len(split[1]) != self.num_orbital_dimensions
        ):
            raise ValueError("Invalid permutation.")

        permutation = f"{split[0]}...->{split[1]}..."

        return LocalFourPoint(
            np.einsum(permutation, self.mat, optimize=True),
            self.channel,
            self.num_wn_dimensions,
            self.num_vn_dimensions,
            self.full_niw_range,
            self.full_niv_range,
        )

    def change_frequency_notation_ph_to_pp(self) -> "LocalFourPoint":
        """
        Changes the frequency notation of the object from ph to pp and returns the object.
        """
        if self.num_wn_dimensions + self.num_vn_dimensions != 3:
            raise ValueError("Only objects with three frequency dimensions are supported.")

        if self.channel != SpinChannel.DENS and self.channel != SpinChannel.MAGN:
            raise ValueError("Only density and magnetic objects are supported.")

        iw_pp, iv_pp, ivp_pp = MFHelper.get_frequencies_for_ph_to_pp_channel_conversion(self.niw, self.niv)

        return LocalFourPoint(
            self[..., iw_pp, iv_pp, ivp_pp],
            self.channel,
            1,
            2,
            full_niw_range=True,
            full_niv_range=True,
            frequency_notation=FrequencyNotation.PP,
        )

    def change_frequency_notation_ph_to_ph_bar(
        self, other: "LocalFourPoint"
    ) -> tuple["LocalFourPoint", "LocalFourPoint"]:
        """
        Changes the frequency notation of the object from ph to ph_bar and returns the object as a tuple in (d, m) channels.
        Requires a density and a magnetic object.
        """
        if (
            self.num_wn_dimensions + self.num_vn_dimensions != 3
            or other.num_vn_dimensions + other.num_wn_dimensions != 3
        ):
            raise ValueError("Only objects with three frequency dimensions are supported.")

        if self.niw != other.niw or self.niv != other.niv:
            raise ValueError("Both objects must have the same number of frequencies.")

        if self.channel == other.channel and self.channel != SpinChannel.NONE:
            raise ValueError(
                "Channel of both objects must be different, one 'density' and one 'magnetic' object are needed."
            )

        iw_ph_bar, iv_ph_bar, ivp_ph_bar = MFHelper.get_frequencies_for_ph_to_ph_bar_channel_conversion(
            self.niw, self.niv
        )

        mat_ph_dens = (self if self.channel == SpinChannel.DENS else other).mat[..., iw_ph_bar, iv_ph_bar, ivp_ph_bar]
        mat_ph_magn = (self if self.channel == SpinChannel.MAGN else other).mat[..., iw_ph_bar, iv_ph_bar, ivp_ph_bar]

        return (
            LocalFourPoint(
                -0.5 * mat_ph_dens - 1.5 * mat_ph_magn,
                SpinChannel.DENS,
                1,
                2,
                full_niw_range=True,
                full_niv_range=True,
                frequency_notation=FrequencyNotation.PH_BAR,
            ).permute_orbitals("abcd->cbad"),
            LocalFourPoint(
                -0.5 * (mat_ph_dens - mat_ph_magn),
                SpinChannel.MAGN,
                1,
                2,
                full_niw_range=True,
                full_niv_range=True,
                frequency_notation=FrequencyNotation.PH_BAR,
            ).permute_orbitals("abcd->cbad"),
        )

    def pad_with_u(self, u: LocalInteraction, niv_pad: int):
        """
        Used to pad a LocalFourPoint object with a LocalInteraction object outside of the core frequency region.
        If niv_pad is less or equal self.niv, no padding will be done.
        """
        copy = deepcopy(self)

        if niv_pad <= copy.niv:
            return copy

        gamma_urange_mat = np.tile(
            u.mat[..., None, None, None], (1, 1, 1, 1, 2 * copy.niw + 1, 2 * niv_pad, 2 * niv_pad)
        )
        niv_diff = niv_pad - copy.niv
        gamma_urange_mat[..., niv_diff : niv_diff + 2 * copy.niv, niv_diff : niv_diff + 2 * copy.niv] = copy.mat
        copy.mat = gamma_urange_mat
        copy.update_original_shape()
        return copy

    @staticmethod
    def load(
        filename: str,
        channel: SpinChannel = SpinChannel.NONE,
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ) -> "LocalFourPoint":
        return LocalFourPoint(
            np.load(filename, allow_pickle=False),
            channel,
            num_wn_dimensions,
            num_vn_dimensions,
            full_niw_range,
            full_niv_range,
            frequency_notation,
        )

    @staticmethod
    def from_constant(
        n_bands: int,
        niw: int,
        niv: int,
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        channel: SpinChannel = SpinChannel.NONE,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
        value: float = 0.0,
    ) -> "LocalFourPoint":
        """
        Initializes the object with a constant value.
        """
        shape = (n_bands,) * 4 + (2 * niw + 1,) * num_wn_dimensions + (2 * niv,) * num_vn_dimensions
        return LocalFourPoint(
            np.full(shape, value),
            channel,
            num_wn_dimensions,
            num_vn_dimensions,
            frequency_notation=frequency_notation,
        )

    @staticmethod
    def identity(
        n_bands: int, niw: int, niv: int, num_vn_dimensions: int = 2, full_niw_range: bool = False
    ) -> "LocalFourPoint":
        if num_vn_dimensions not in (1, 2):
            raise ValueError("Invalid number of fermionic frequency dimensions.")
        full_shape = (n_bands,) * 4 + (2 * niw + 1,) + (2 * niv,) * num_vn_dimensions
        compound_index_size = 2 * niv * n_bands**2
        mat = np.tile(np.eye(compound_index_size)[None, ...], (2 * niw + 1, 1, 1))

        result = LocalFourPoint(
            mat, num_vn_dimensions=num_vn_dimensions, full_niw_range=full_niw_range
        ).to_full_indices(full_shape)
        if num_vn_dimensions == 1:
            return result.take_vn_diagonal()
        return result

    @staticmethod
    def identity_like(other: "LocalFourPoint") -> "LocalFourPoint":
        return LocalFourPoint.identity(
            other.n_bands, other.niw, other.niv, other.num_vn_dimensions, other.full_niw_range
        )

    def plot(
        self,
        orbs: np.ndarray | list | tuple = (0, 0, 0, 0),
        omega: int = 0,
        do_save: bool = True,
        output_dir: str = "./",
        name: str = "Name",
        colormap: str = "RdBu",
        show: bool = False,
    ) -> None:
        """
        Plots the four-point object for a given set of orbitals and a given bosonic frequency.
        """
        if np.abs(omega) > self.niw:
            raise ValueError(f"Omega {omega} out of range.")
        if len(orbs) != 4:
            raise ValueError("'orbs' needs to be an array of size 4.")

        fig, axes = plt.subplots(ncols=2, figsize=(7, 3), dpi=251)
        axes = axes.flatten()
        wn_list = MFHelper.wn(self.niw)
        wn_index = np.argmax(wn_list == omega)
        mat = self.mat[orbs[0], orbs[1], orbs[2], orbs[3], wn_index, ...]
        vn = MFHelper.vn(self.niv)
        im1 = axes[0].pcolormesh(vn, vn, mat.real, cmap=colormap)
        im2 = axes[1].pcolormesh(vn, vn, mat.imag, cmap=colormap)
        axes[0].set_title(r"$\Re$")
        axes[1].set_title(r"$\Im$")
        for ax in axes:
            ax.set_xlabel(r"$\nu_p$")
            ax.set_ylabel(r"$\nu$")
            ax.set_aspect("equal")
        fig.suptitle(name)
        fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location="right", pad=0.05)
        fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location="right", pad=0.05)
        plt.tight_layout()
        if do_save:
            plt.savefig(output_dir + "/" + name + f"_w{omega}" + ".png")
        if show:
            plt.show()
        else:
            plt.close()

    def _revert_frequency_dimensions_after_operation(
        self, other: "LocalFourPoint", other_extended: bool, self_extended: bool
    ):
        if self_extended:
            self.take_vn_diagonal()
        if other_extended:
            other = other.take_vn_diagonal()
        return other
