import numpy as np
import matplotlib.pyplot as plt
from matsubara_frequency_helper import MFHelper

from local_n_point import LocalNPoint
from local_three_point import LocalThreePoint
from local_two_point import LocalTwoPoint


class LocalFourPoint(LocalNPoint):
    def __init__(self, mat: np.ndarray, channel: str, full_niw_range: bool = True, full_niv_range: bool = True):
        super().__init__(mat, 4, 3, full_niv_range)
        self._channel = channel
        self._full_niw_range = full_niw_range

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def niw(self) -> int:
        return self.original_shape[-3] // 2 if self.full_niv_range else self.original_shape[-3]

    @property
    def full_niw_range(self) -> bool:
        return self._full_niw_range

    def cut_niw(self, niw_cut: int) -> "LocalFourPoint":
        if niw_cut > self.niw:
            raise ValueError("Cannot cut more bosonic frequencies than the object has.")

        if self.full_niv_range:
            self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1, :, :]
        else:
            self.mat = self.mat[..., :niw_cut, :, :]

        self.original_shape = self.mat.shape
        return self

    def cut_niv(self, niv_cut: int) -> "LocalFourPoint":
        if niv_cut > self.niv:
            raise ValueError("Cannot cut more fermionic frequencies than the object has.")

        if self.full_niv_range:
            self.mat = self.mat[..., self.niv - niv_cut : self.niv + niv_cut, self.niv - niv_cut : self.niv + niv_cut]
        else:
            self.mat = self.mat[..., :niv_cut, :niv_cut]

        self.original_shape = self.mat.shape
        return self

    def cut_niw_and_niv(self, niw_cut: int, niv_cut: int) -> "LocalFourPoint":
        return self.cut_niw(niw_cut).cut_niv(niv_cut)

    def symmetrize_v_vp(self) -> "LocalFourPoint":
        self.mat = 0.5 * (self.mat + np.swapaxes(self.mat, -1, -2))
        return self

    def invert(self) -> "LocalFourPoint":
        copy = self
        copy = copy.to_compound_indices()
        copy.mat = np.linalg.inv(copy.mat)
        return copy.to_full_indices()

    def to_compound_indices(self) -> "LocalFourPoint":
        if len(self.current_shape) == 3:  # [w,x1,x2]
            return self
        elif len(self.current_shape) == 7:  # [o1,o2,o3,o4,w,v,vp]
            self.original_shape = self.mat.shape
            self.mat = np.ascontiguousarray(
                self.mat.transpose(4, 0, 1, 5, 2, 3, 6).reshape(
                    2 * self.niw + 1, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
                )
            )
            return self
        else:
            raise ValueError(f"Converting to compound indices with shape {self.current_shape} not supported.")

    def to_full_indices(self, shape: tuple = None) -> "LocalFourPoint":
        if len(self.current_shape) == 7:  # [o1,o2,o3,o4,w,v,vp]
            return self
        elif len(self.current_shape) == 3:  # [w,x1,x2]
            self.original_shape = shape if shape is not None else self.original_shape
            self.mat = np.ascontiguousarray(
                self.mat.reshape(
                    2 * self.niw + 1, self.n_bands, self.n_bands, 2 * self.niv, self.n_bands, self.n_bands, 2 * self.niv
                ).transpose(1, 2, 4, 5, 0, 3, 6)
            )
            return self
        else:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")

    def _matmul_core(self, other: LocalNPoint, left_hand_side: bool = True) -> LocalNPoint:
        if not isinstance(other, (LocalTwoPoint, LocalThreePoint, LocalFourPoint)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, (LocalThreePoint, LocalFourPoint)):
            if self.channel != other.channel:
                raise ValueError("Channels don't match.")
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
            return LocalTwoPoint(new_mat, self.full_niv_range).to_full_indices(other.current_shape)

        if isinstance(other, LocalThreePoint):
            return LocalThreePoint(new_mat, self.channel, self.full_niw_range, self.full_niw_range).to_full_indices(
                other.current_shape
            )

        return LocalFourPoint(new_mat, self.channel, self.full_niw_range, self.full_niv_range).to_full_indices(
            other.current_shape
        )

    def __matmul__(self, other: LocalNPoint) -> LocalNPoint:
        return self._matmul_core(other, left_hand_side=True)

    def __rmatmul__(self, other: LocalNPoint) -> LocalNPoint:
        return self._matmul_core(other, left_hand_side=False)

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
            plt.savefig(target_directory + "/" + figure_name + ".png")
        if show:
            plt.show()
        else:
            plt.close()
