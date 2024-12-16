import numpy as np
from matplotlib import pyplot as plt

from i_have_channel import IHaveChannel, Channel
from interaction import LocalInteraction
from local_n_point import LocalNPoint
from local_three_point import LocalThreePoint
from local_two_point import LocalTwoPoint
from matsubara_frequencies import MFHelper


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
        IHaveChannel.__init__(self, channel)

    def __matmul__(self, other) -> LocalNPoint:
        return self._execute_matmul(other, left_hand_side=True)

    def __rmatmul__(self, other) -> LocalNPoint:
        return self._execute_matmul(other, left_hand_side=False)

    def __add__(self, other) -> "LocalFourPoint":
        return self._execute_add_sub(other, is_addition=True)

    def __radd__(self, other) -> "LocalFourPoint":
        return self.__add__(other)

    def __sub__(self, other) -> "LocalFourPoint":
        return self._execute_add_sub(other, is_addition=False)

    def __rsub__(self, other) -> "LocalFourPoint":
        return self.__sub__(other)

    def _execute_matmul(self, other, left_hand_side: bool = True) -> LocalNPoint:
        """
        Helper method that allows for matrix multiplication for LocalFourPoint objects. Depending on the
        number of frequency dimensions, the objects have to be multiplied differently.
        """
        if not isinstance(other, (LocalTwoPoint, LocalThreePoint, LocalFourPoint)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, (LocalThreePoint, LocalFourPoint)):
            if self.niw != other.niw:
                raise ValueError(
                    f"Shapes {self.current_shape} and {other.current_shape} do not match for multiplication!"
                )

        if self.niv != other.niv or self.n_bands != other.n_bands:
            raise ValueError(f"Shapes {self.current_shape} and {other.current_shape} do not match for multiplication!")

        self.to_compound_indices()
        other = other.to_compound_indices()
        # for __matmul__ self needs to be the LHS object, for __rmatmul__ self needs to be the RHS object
        new_mat = np.matmul(self.mat, other.mat) if left_hand_side else np.matmul(other.mat, self.mat)
        self.to_full_indices()
        other = other.to_full_indices()

        if isinstance(other, LocalTwoPoint):
            # TODO: or np.mean?
            new_mat = np.sum(new_mat, -2)
            return LocalTwoPoint(new_mat, full_niv_range=self.full_niv_range).to_full_indices(
                self.original_shape if self.num_fermionic_frequency_dimensions == 0 else other.original_shape
            )

        channel = self.channel if self.channel != Channel.NONE else other.channel

        if isinstance(other, LocalThreePoint):
            return LocalThreePoint(
                new_mat, channel, 1, 1, full_niw_range=self.full_niw_range, full_niv_range=self.full_niw_range
            ).to_full_indices(
                self.original_shape if self.num_fermionic_frequency_dimensions == 1 else other.original_shape
            )

        return LocalFourPoint(
            new_mat, channel, 1, 2, full_niw_range=self.full_niw_range, full_niv_range=self.full_niv_range
        ).to_full_indices(self.original_shape if self.num_fermionic_frequency_dimensions == 2 else other.original_shape)

    def _execute_add_sub(self, other, is_addition: bool = True) -> "LocalFourPoint":
        """
        Helper method that allows for in-place addition and subtraction for LocalFourPoint objects. Depending on the
        number of frequency dimensions, the objects have to be added differently.
        """
        if not isinstance(other, (LocalInteraction, LocalFourPoint, LocalThreePoint)):
            raise ValueError(f"Operations '+/-' for {type(self)} and {type(other)} not supported.")

        channel = self.channel if self.channel != Channel.NONE else other.channel

        if isinstance(other, (LocalFourPoint, LocalThreePoint)):
            if self.niw != other.niw or self.niv != other.niv or self.n_bands != other.n_bands:
                raise ValueError(f"Shapes {self.current_shape} and {other.current_shape} do not match!")
            if self.num_bosonic_frequency_dimensions != other.num_bosonic_frequency_dimensions:
                raise ValueError("Number of bosonic frequency dimensions do not match.")

        self_mat, other_mat = self.mat, other.mat

        if isinstance(other, LocalThreePoint):
            other_mat = np.einsum("...i,ij->...ij", other_mat, np.eye(other_mat.shape[-1]), optimize=True)

        if len(self_mat.shape) != len(other_mat.shape):
            diff = len(self_mat.shape) - len(other_mat.shape)
            if diff > 0:
                other_mat = np.reshape(other_mat, other_mat.shape + (1,) * diff)
            else:
                self_mat = np.reshape(self_mat, self_mat.shape + (1,) * -diff)

        if (
            isinstance(other, LocalInteraction)
            and self.num_bosonic_frequency_dimensions == 1
            and self.num_fermionic_frequency_dimensions == 0
        ):
            np.add(self_mat, other_mat, out=self_mat) if is_addition else np.subtract(self_mat, other_mat, out=self_mat)
            return LocalFourPoint(self_mat, channel, 1, 0, self.full_niw_range, self.full_niv_range)

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

        self.mat = np.einsum(f"{split[0]}...->{split[1]}...", self.mat, optimize=True)
        diff = len(split[0]) - len(split[1])
        self.original_shape = self.current_shape
        self._num_orbital_dimensions = self.num_orbital_dimensions - diff
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
            copy_mat, self.channel, 1, remaining_fermionic_dimensions, self.full_niw_range, self.full_niv_range
        )

    def contract_legs(self, beta: float) -> "LocalFourPoint":
        """
        Sums over all fermionic frequency dimensions if the object has 2 fermionic frequency dimensions.
        """
        if self.num_fermionic_frequency_dimensions != 2:
            raise ValueError("This method is only implemented for 2 fermionic frequency dimensions.")
        return self.sum_over_fermionic_dimensions(beta, axis=(-1, -2))

    def permute_orbitals(self, permutation: str = "ijkl->ijkl") -> "LocalFourPoint":
        """
        Permutes the orbitals of the four-point object.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        permutation = f"{split[0]}...->{split[1]}..."

        return LocalFourPoint(
            np.einsum(permutation, self.mat, optimize=True),
            self.channel,
            self.num_bosonic_frequency_dimensions,
            self.num_fermionic_frequency_dimensions,
            self.full_niw_range,
            self.full_niv_range,
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
