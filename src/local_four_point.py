from matplotlib import pyplot as plt

from interaction import LocalInteraction
from local_n_point import LocalNPoint
from local_three_point import LocalThreePoint
from matsubara_frequencies import MFHelper
from n_point_base import *


class LocalFourPoint(LocalNPoint, IHaveChannel):
    """
    This class is used to represent a local four-point object in a given channel with a given number of bosonic and
    fermionic frequency dimensions that have to be added to keep track of (re-)shaping.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: Channel = Channel.NONE,
        num_bosonic_frequency_dimensions: int = 1,
        num_fermionic_frequency_dimensions: int = 2,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ):
        LocalNPoint.__init__(
            self,
            mat,
            4,
            num_bosonic_frequency_dimensions,
            num_fermionic_frequency_dimensions,
            full_niw_range,
            full_niv_range,
        )
        IHaveChannel.__init__(self, channel, frequency_notation)

    def __matmul__(self, other: "LocalFourPoint") -> "LocalFourPoint":
        return self._execute_matmul(other, left_hand_side=True)

    def __rmatmul__(self, other: "LocalFourPoint") -> "LocalFourPoint":
        return self._execute_matmul(other, left_hand_side=False)

    def __add__(self, other) -> "LocalFourPoint":
        return self._execute_add_sub(other, is_addition=True)

    def __radd__(self, other) -> "LocalFourPoint":
        return self.__add__(other)

    def __sub__(self, other) -> "LocalFourPoint":
        return self._execute_add_sub(other, is_addition=False)

    def __rsub__(self, other) -> "LocalFourPoint":
        return self.__sub__(other)

    def _execute_matmul(self, other: "LocalFourPoint", left_hand_side: bool = True) -> "LocalFourPoint":
        """
        Helper method that allows for matrix multiplication for LocalFourPoint objects. Depending on the
        number of frequency dimensions, the objects have to be multiplied differently.
        """
        if not isinstance(other, LocalFourPoint):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if self.niw != other.niw or self.niv != other.niv or self.n_bands != other.n_bands:
            raise ValueError(f"Shapes {self.current_shape} and {other.current_shape} do not match for multiplication!")

        self.to_compound_indices()
        other = other.to_compound_indices()
        # for __matmul__ self needs to be the LHS object, for __rmatmul__ self needs to be the RHS object
        new_mat = np.matmul(self.mat, other.mat) if left_hand_side else np.matmul(other.mat, self.mat)
        self.to_full_indices()
        other = other.to_full_indices()

        channel = self.channel if self.channel != Channel.NONE else other.channel

        return LocalFourPoint(
            new_mat, channel, 1, 2, full_niw_range=self.full_niw_range, full_niv_range=self.full_niv_range
        ).to_full_indices(self.original_shape if self.num_fermionic_frequency_dimensions == 2 else other.original_shape)

    def _execute_add_sub(self, other, is_addition: bool = True) -> "LocalFourPoint":
        """
        Helper method that allows for in-place addition and subtraction for LocalFourPoint objects. Depending on the
        number of frequency dimensions, the objects have to be added differently.
        """
        if not isinstance(other, (LocalInteraction, LocalFourPoint, LocalThreePoint, np.ndarray)):
            raise ValueError(f"Operations '+/-' for {type(self)} and {type(other)} not supported.")

        if isinstance(other, (LocalFourPoint, LocalThreePoint)):
            if self.niw != other.niw or self.niv != other.niv or self.n_bands != other.n_bands:
                raise ValueError(f"Shapes {self.current_shape} and {other.current_shape} do not match!")
            if self.num_bosonic_frequency_dimensions != other.num_bosonic_frequency_dimensions:
                raise ValueError("Number of bosonic frequency dimensions do not match.")

        self_mat, other_mat = self.mat, other.mat
        channel = self.channel if self.channel != Channel.NONE else other.channel

        if isinstance(other, LocalInteraction):
            if self.num_bosonic_frequency_dimensions == 1 and self.num_fermionic_frequency_dimensions == 0:
                other_mat = other_mat.reshape(other_mat.shape + (1,))
            elif self.num_bosonic_frequency_dimensions == 1 and self.num_fermionic_frequency_dimensions == 2:
                other_mat = (
                    other_mat.reshape(other_mat.shape + (1,) * 3)
                    * np.eye(2 * self.niv)[None, None, None, None, None, :, :]
                )

            np.add(self_mat, other_mat, out=self_mat) if is_addition else np.subtract(self_mat, other_mat, out=self_mat)
            return LocalFourPoint(self_mat, channel, 1, 2, self.full_niw_range, self.full_niv_range)

        if other.num_fermionic_frequency_dimensions == 1:
            other_mat = np.einsum("...i,ij->...ij", other_mat, np.eye(other_mat.shape[-1]))

        np.add(self_mat, other_mat, out=self_mat) if is_addition else np.subtract(self_mat, other_mat, out=self_mat)
        if self.num_fermionic_frequency_dimensions == 0 and other.num_fermionic_frequency_dimensions == 0:
            return LocalFourPoint(self_mat, channel, 1, 0, self.full_niw_range, self.full_niv_range)
        return LocalFourPoint(self_mat, channel, 1, 2, self.full_niw_range, self.full_niv_range)

    def symmetrize_v_vp(self) -> "LocalFourPoint":
        """
        Symmetrize with respect to (v,v'). This is justified for SU(2) symmetric systems. (Thesis Rohringer p. 72)
        """
        self.mat = 0.5 * (self.mat + np.swapaxes(self.mat, -1, -2))
        return self

    def sum_over_orbitals(self, orbital_contraction: str = "abcd->ad") -> "LocalFourPoint":
        """
        Sums over the given orbitals.
        """
        split = orbital_contraction.split("->")
        if len(split[0]) != 4 or len(split[1]) > len(split[0]):
            raise ValueError("Invalid orbital contraction.")

        self.mat = np.einsum(f"{split[0]}...->{split[1]}...", self.mat)
        diff = len(split[0]) - len(split[1])
        self.original_shape = self.current_shape
        self._num_orbital_dimensions -= diff
        return self

    def sum_over_fermionic_dimensions(self, beta: float, axis: tuple = (-1,)) -> "LocalFourPoint":
        """
        This method is used to sum over specific fermionic frequency dimensions and multiplies with the correct prefactor 1/beta^(n_dim).
        """
        if len(axis) > self.num_fermionic_frequency_dimensions:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        remaining_fermionic_dimensions = self.num_fermionic_frequency_dimensions - len(axis)
        copy_mat = 1 / beta ** len(axis) * np.sum(self.mat, axis=axis)
        return LocalFourPoint(
            copy_mat,
            self.channel,
            self.num_bosonic_frequency_dimensions,
            remaining_fermionic_dimensions,
            self.full_niw_range,
            self.full_niv_range,
        )

    def contract_legs(self, beta: float) -> "LocalFourPoint":
        """
        Sums over all fermionic frequency dimensions if the object has 2 fermionic frequency dimensions.
        """
        if self.num_fermionic_frequency_dimensions != 2:
            raise ValueError("This method is only implemented for 2 fermionic frequency dimensions.")
        return self.sum_over_fermionic_dimensions(beta, axis=(-1, -2)).sum_over_orbitals("abcd->ad")

    def permute_orbitals(self, permutation: str = "ijkl->ijkl") -> "LocalFourPoint":
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
            np.einsum(permutation, self.mat),
            self.channel,
            self.num_bosonic_frequency_dimensions,
            self.num_fermionic_frequency_dimensions,
            self.full_niw_range,
            self.full_niv_range,
        )

    def change_frequency_notation_ph_to_pp(self) -> "LocalFourPoint":
        """
        Changes the frequency notation of the object from ph to pp and returns the object.
        """
        if self.num_bosonic_frequency_dimensions + self.num_fermionic_frequency_dimensions != 3:
            raise ValueError("Only objects with three frequency dimensions are supported.")

        if self.channel != Channel.DENS and self.channel != Channel.MAGN:
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
            self.num_bosonic_frequency_dimensions + self.num_fermionic_frequency_dimensions != 3
            or other.num_fermionic_frequency_dimensions + other.num_bosonic_frequency_dimensions != 3
        ):
            raise ValueError("Only objects with three frequency dimensions are supported.")

        if self.niw != other.niw or self.niv != other.niv:
            raise ValueError("Both objects must have the same number of frequencies.")

        if self.channel == other.channel and self.channel != Channel.NONE:
            raise ValueError(
                "Channel of both objects must be different, one 'density' and one 'magnetic' object are needed."
            )

        iw_ph_bar, iv_ph_bar, ivp_ph_bar = MFHelper.get_frequencies_for_ph_to_ph_bar_channel_conversion(
            self.niw, self.niv
        )

        mat_ph_dens = (self if self.channel == Channel.DENS else other).mat[..., iw_ph_bar, iv_ph_bar, ivp_ph_bar]
        mat_ph_magn = (self if self.channel == Channel.MAGN else other).mat[..., iw_ph_bar, iv_ph_bar, ivp_ph_bar]

        return (
            LocalFourPoint(
                0.5 * mat_ph_dens + 1.5 * mat_ph_magn,
                self.channel,
                1,
                2,
                full_niw_range=True,
                full_niv_range=True,
                frequency_notation=FrequencyNotation.PH_BAR,
            ),
            LocalFourPoint(
                0.5 * (mat_ph_dens - mat_ph_magn),
                self.channel,
                1,
                2,
                full_niw_range=True,
                full_niv_range=True,
                frequency_notation=FrequencyNotation.PH_BAR,
            ),
        )

    def concatenate_local_u(self, u_loc: LocalInteraction, niv_full: int) -> "LocalFourPoint":
        pad_size = niv_full - self.niv

        if pad_size <= 0:
            return self

        mat_padded = np.broadcast_to(
            u_loc.mat[..., None, None, None], (self.n_bands,) * 4 + (2 * self.niw + 1,) + (2 * niv_full,) * 2
        ).copy()
        mat_padded[..., pad_size : pad_size + 2 * self.niv, pad_size : pad_size + 2 * self.niv] = self.mat
        return LocalFourPoint(
            mat_padded, self.channel, self.num_bosonic_frequency_dimensions, self.num_fermionic_frequency_dimensions
        )

    def plot(
        self,
        orbs: np.ndarray | list = [0, 0, 0, 0],
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
        vn = MFHelper.vn(self.niv)  # pylint: disable=unexpected-keyword-arg
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
