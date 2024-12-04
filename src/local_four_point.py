import numpy as np
from matplotlib import pyplot as plt

from i_have_channel import IHaveChannel, Channel
from interaction import LocalInteraction
from local_n_point import LocalNPoint
from local_three_point import LocalThreePoint
from local_two_point import LocalTwoPoint
from matsubara_frequency_helper import MFHelper


class LocalFourPoint(LocalNPoint, IHaveChannel):
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
        if not isinstance(other, (LocalTwoPoint, LocalThreePoint, LocalFourPoint)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, (LocalThreePoint, LocalFourPoint)):
            if self.channel != other.channel:
                raise ValueError(f"Channels {self.channel} and {other.channel} don't match.")
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
            new_mat = np.sum(new_mat, -2)
            return LocalTwoPoint(new_mat, full_niv_range=self.full_niv_range).to_full_indices(other.current_shape)

        if isinstance(other, LocalThreePoint):
            return LocalThreePoint(
                new_mat, self.channel, full_niw_range=self.full_niw_range, full_niv_range=self.full_niw_range
            ).to_full_indices(other.current_shape)

        return LocalFourPoint(
            new_mat, self.channel, full_niw_range=self.full_niw_range, full_niv_range=self.full_niv_range
        ).to_full_indices(other.current_shape)

    def _execute_add_sub(self, other, is_addition: bool = True) -> "LocalFourPoint":
        if not isinstance(other, (LocalInteraction, LocalFourPoint, LocalThreePoint)):
            raise ValueError(f"Operation for {type(self)} and {type(other)} not supported.")

        self_mat, other_mat = self.mat, other.mat

        if isinstance(other, (LocalFourPoint, LocalThreePoint)):
            if self.channel != other.channel:
                raise ValueError(f"Channels {self.channel} and {other.channel} don't match.")
            if self.niw != other.niw or self.niv != other.niv or self.n_bands != other.n_bands:
                raise ValueError(f"Shapes {self.current_shape} and {other.current_shape} do not match!")
            if self.num_bosonic_frequency_dimensions != other.num_bosonic_frequency_dimensions:
                raise ValueError("Number of bosonic frequency dimensions do not match.")

        if isinstance(other, LocalThreePoint):
            other_mat = MFHelper.extend_last_frequency_axis_to_diagonal(other_mat)

        if isinstance(other, LocalInteraction):
            if self.num_bosonic_frequency_dimensions == 1:
                other_mat = other_mat * np.eye(2 * self.niv)[None, None, None, None, None, :, :]

        return LocalFourPoint(
            self_mat + other_mat if is_addition else self_mat - other_mat,
            self.channel,
            self.full_niw_range,
            self.full_niv_range,
        )

    def symmetrize_v_vp(self) -> "LocalFourPoint":
        self.mat = 0.5 * (self.mat + np.swapaxes(self.mat, -1, -2))
        return self

    def contract_legs(self) -> "LocalFourPoint":
        copy_mat = np.sum(self.mat, axis=(-1, -2))
        return LocalFourPoint(copy_mat, self.channel, 1, 0, self.full_niw_range, self.full_niv_range)

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
