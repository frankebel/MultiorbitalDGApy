import numpy as np
from matplotlib import pyplot as plt

from i_have_channel import IHaveChannel, Channel
from interaction import LocalInteraction
from local_n_point import LocalNPoint
from local_three_point import LocalThreePoint
from local_two_point import LocalTwoPoint
from matsubara_frequency_helper import MFHelper


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

        if isinstance(other, LocalThreePoint):
            return LocalThreePoint(
                new_mat, self.channel, 1, 1, full_niw_range=self.full_niw_range, full_niv_range=self.full_niw_range
            ).to_full_indices(
                self.original_shape if self.num_fermionic_frequency_dimensions == 1 else other.original_shape
            )

        return LocalFourPoint(
            new_mat, self.channel, 1, 2, full_niw_range=self.full_niw_range, full_niv_range=self.full_niv_range
        ).to_full_indices(self.original_shape if self.num_fermionic_frequency_dimensions == 2 else other.original_shape)

    def _execute_add_sub(self, other, is_addition: bool = True) -> "LocalFourPoint":
        """
        Helper method that allows for in-place addition and subtraction for LocalFourPoint objects. Depending on the
        number of frequency dimensions, the objects have to be added differently.
        """
        if not isinstance(other, (LocalInteraction, LocalFourPoint, LocalThreePoint)):
            raise ValueError(f"Operation for {type(self)} and {type(other)} not supported.")

        self_mat, other_mat = self.mat, other.mat

        if isinstance(other, (LocalFourPoint, LocalThreePoint)):
            if self.niw != other.niw or self.niv != other.niv or self.n_bands != other.n_bands:
                raise ValueError(f"Shapes {self.current_shape} and {other.current_shape} do not match!")
            if self.num_bosonic_frequency_dimensions != other.num_bosonic_frequency_dimensions:
                raise ValueError("Number of bosonic frequency dimensions do not match.")

        if isinstance(other, LocalThreePoint):
            other_mat = MFHelper.extend_last_frequency_axis_to_diagonal(other_mat)

        if isinstance(other, LocalInteraction):
            if self.num_bosonic_frequency_dimensions == 1 and self.num_fermionic_frequency_dimensions == 0:
                result_mat = self_mat + other_mat if is_addition else self_mat - other_mat
                return LocalFourPoint(result_mat, self.channel, 1, 0, self.full_niw_range, self.full_niv_range)
            if self.num_bosonic_frequency_dimensions == 1 and self.num_fermionic_frequency_dimensions == 2:
                other_mat = other_mat * np.eye(2 * self.niv)[None, None, None, None, None, :, :]

        result_mat = self_mat + other_mat if is_addition else self_mat - other_mat
        if self.num_fermionic_frequency_dimensions == 0 and other.num_fermionic_frequency_dimensions == 0:
            return LocalFourPoint(result_mat, self.channel, 1, 0, self.full_niw_range, self.full_niv_range)
        return LocalFourPoint(result_mat, self.channel, 1, 2, self.full_niw_range, self.full_niv_range)

    def symmetrize_v_vp(self) -> "LocalFourPoint":
        self.mat = 0.5 * (self.mat + np.swapaxes(self.mat, -1, -2))
        return self

    def sum_over_fermionic_dimensions(self, axis: tuple = (-1,)) -> "LocalFourPoint":
        """
        This method is used to sum over specific fermionic frequency dimensions and returns the new objects with
        the new shape.
        """
        if len(axis) > self.num_fermionic_frequency_dimensions:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        remaining_fermionic_dimensions = self.num_fermionic_frequency_dimensions - len(axis)
        copy_mat = np.sum(self.mat, axis=axis)
        return LocalFourPoint(
            copy_mat, self.channel, 1, remaining_fermionic_dimensions, self.full_niw_range, self.full_niv_range
        )

    def contract_legs(self) -> "LocalFourPoint":
        """
        Sums over all fermionic frequency dimensions.
        """
        return self.sum_over_fermionic_dimensions(axis=(-1, -2))

    def plot(
        self,
        orbs: np.ndarray | list = [0, 0, 0, 0],
        omega: int = 0,
        do_save: bool = True,
        target_directory: str = "./",
        figure_name: str = "Name",
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
        wn_list = MFHelper.get_wn_int(self.niw)
        wn_index = np.argmax(wn_list == omega)
        mat = self.mat[orbs[0], orbs[1], orbs[2], orbs[3], wn_index, ...]
        vn = MFHelper.get_vn_int(self.niv)  # pylint: disable=unexpected-keyword-arg
        im1 = axes[0].pcolormesh(vn, vn, mat.real, cmap=colormap)
        im2 = axes[1].pcolormesh(vn, vn, mat.imag, cmap=colormap)
        axes[0].set_title(r"$\Re$")
        axes[1].set_title(r"$\Im$")
        for ax in axes:
            ax.set_xlabel(r"$\nu_p$")
            ax.set_ylabel(r"$\nu$")
            ax.set_aspect("equal")
        fig.suptitle(figure_name)
        fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location="right", pad=0.05)
        fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location="right", pad=0.05)
        plt.tight_layout()
        if do_save:
            plt.savefig(target_directory + "/" + figure_name + f"_w{omega}" + ".png")
        if show:
            plt.show()
        else:
            plt.close()
